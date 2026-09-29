import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2, numpy as np

# 复用你昨天写好的函数
def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0, None
    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
    if not valid_contours: return 0.0, None
    c = max(valid_contours, key=cv2.contourArea)
    return cv2.contourArea(c), c

# --- 核心测试函数 ---
def run_test(img_path, real_area, label):
    img = cv2.imread(img_path)
    if img is None:
        print(f"【{label}】图片读取失败：{img_path}")
        return
        
    # 中值滤波 + HSV
    img = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 绿色叶片阈值 (沿用你调好的)
    leaf_mask = cv2.inRange(hsv, np.array([35, 25, 15]), np.array([85, 255, 255]))
    # 蓝色方块阈值
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 20]), np.array([140, 255, 255]))
    
    # 形态学去噪
    k_open, k_close = np.ones((7, 7), np.uint8), np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
    blue_mask = cv2.morphologyEx(cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
    
    # 面积提取
    leaf_px, _ = largest_contour(leaf_mask)
    ref_px, ref_c = largest_contour(blue_mask)
    
    # 凸包修复蓝方块
    if ref_c is not None:
        ref_px = cv2.contourArea(cv2.convexHull(ref_c))
        
    print(f"--- {label} 测试结果 ---")
    print(f"参照物真实面积: {real_area} cm² | 参照物像素面积: {ref_px:.1f} px²")
    print(f"叶片像素面积: {leaf_px:.1f} px²")
    
    if ref_px > 0:
        scale = real_area / ref_px
        leaf_real = leaf_px * scale
        print(f"比例系数: {scale:.8f} | 预测叶片面积: {leaf_real:.3f} cm²\n")
    else:
        print("方块识别失败！\n")

# ========== 开始测试 ==========
# 1. 定标：只读取2cm图片，算出唯一的比例系数
img_2cm = cv2.imread("images/test_2cm.jpg")
img_2cm = cv2.medianBlur(img_2cm, 5)
hsv_2cm = cv2.cvtColor(img_2cm, cv2.COLOR_BGR2HSV)
mask_ref = cv2.inRange(hsv_2cm, np.array([90, 40, 20]), np.array([140, 255, 255]))
mask_ref = cv2.morphologyEx(cv2.morphologyEx(mask_ref, cv2.MORPH_OPEN, np.ones((7,7),np.uint8)), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
ref_px, ref_c = largest_contour(mask_ref)
if ref_c is not None:
    ref_px = cv2.contourArea(cv2.convexHull(ref_c))

# 锁定唯一的、不可更改的比例系数！
ratio = 4.0 / ref_px 
print(f"【锁定比例系数】2cm方块定标：ratio = {ratio:.8f} cm²/px\n")

# 2. 验证函数：用刚才固定的 ratio，去预测别的方块和叶片
def verify_block(img_path, real_area, label, ratio):
    img = cv2.imread(img_path)
    if img is None: 
        print(f"【{label}】图片读取失败")
        return
        
    img = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 提取方块
    mask_ref = cv2.inRange(hsv, np.array([90, 40, 20]), np.array([140, 255, 255]))
    mask_ref = cv2.morphologyEx(cv2.morphologyEx(mask_ref, cv2.MORPH_OPEN, np.ones((7,7),np.uint8)), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    block_px, block_c = largest_contour(mask_ref)
    if block_c is not None:
        block_px = cv2.contourArea(cv2.convexHull(block_c))
        
    # 提取叶片
    mask_leaf = cv2.inRange(hsv, np.array([35, 25, 15]), np.array([85, 255, 255]))
    mask_leaf = cv2.morphologyEx(cv2.morphologyEx(mask_leaf, cv2.MORPH_OPEN, np.ones((7,7),np.uint8)), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    leaf_px, leaf_c = largest_contour(mask_leaf)

    # 用锁定的 ratio 来计算预测面积
    predicted_block_area = block_px * ratio
    predicted_leaf_area = leaf_px * ratio
    block_error = abs(predicted_block_area - real_area) / real_area * 100
    
    print(f"【{label}】")
    print(f"  -> 方块预测面积: {predicted_block_area:.2f} cm² (真值: {real_area} cm²) | 误差: {block_error:.2f}%")
    print(f"  -> 叶片预测面积: {predicted_leaf_area:.2f} cm² (像素: {leaf_px:.1f})\n")

# ========== 重新标定光照测试组 ==========
# 1. 用光照组的"正常光"重新算一个属于这个机位的比例尺
img_light_normal = cv2.imread("images/light_1_normal.jpg") # 换成你的实际图片名
img_light_normal = cv2.medianBlur(img_light_normal, 5)
hsv_light = cv2.cvtColor(img_light_normal, cv2.COLOR_BGR2HSV)

# 提取正常光下的方块
mask_ref_light = cv2.inRange(hsv_light, np.array([90, 40, 20]), np.array([140, 255, 255]))
mask_ref_light = cv2.morphologyEx(cv2.morphologyEx(mask_ref_light, cv2.MORPH_OPEN, np.ones((7,7),np.uint8)), cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
ref_px_light, ref_c_light = largest_contour(mask_ref_light)
if ref_c_light is not None:
    ref_px_light = cv2.contourArea(cv2.convexHull(ref_c_light))

# 新的锁定比例尺！专门用于这组光照测试
ratio_light = 4.0 / ref_px_light
print(f"【光照组重标定】正常光图片的比例尺 ratio_light = {ratio_light:.8f} cm²/px\n")

# 2. 用新比例尺验证其他光照
verify_block("images/light_1_normal.jpg", 4.0, "场景1：正常光", ratio_light)
verify_block("images/light_2_side.jpg", 4.0, "场景2：单侧光", ratio_light)
verify_block("images/light_3_dark.jpg", 4.0, "场景3：暗光", ratio_light)