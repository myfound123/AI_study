import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import matplotlib.pyplot as plt
# 中文补丁
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False   

# ========== 工具函数 ==========
def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0, None
    # 过滤掉面积小于200像素的噪点
    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
    if not valid_contours: return 0.0, None
    c = max(valid_contours, key=cv2.contourArea)
    return cv2.contourArea(c), c

# ========== 主程序 ==========
img_path = "images/test_2cm.jpg"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"找不到图片，检查路径：{img_path}")

# 1. 中值滤波去噪（解决背景散点）
img = cv2.medianBlur(img, 5)  # 直接覆盖 img
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. 绿色叶片阈值（两段合并）
lower_r1, upper_r1 = np.array([35, 25, 15]),   np.array([60, 255, 255])
lower_r2, upper_r2 = np.array([60, 40, 40]), np.array([85, 255, 255])
green_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1), cv2.inRange(hsv, lower_r2, upper_r2))

# 3. 蓝色方块阈值
blue_mask = cv2.inRange(hsv, np.array([90, 40, 20]), np.array([140, 255, 255]))

# 4. 形态学去噪
k_open, k_close = np.ones((7, 7), np.uint8), np.ones((5, 5), np.uint8)
green_mask = cv2.morphologyEx(cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
blue_mask = cv2.morphologyEx(cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)

# 5. 提取轮廓与面积计算
leaf_px, leaf_c = largest_contour(green_mask)
ref_px, ref_c = largest_contour(blue_mask)

# 6. 凸包修复蓝方块反光缺角
if ref_c is not None:
    ref_px = cv2.contourArea(cv2.convexHull(ref_c))

print(f"绿叶像素面积: {leaf_px:.1f} px²")
print(f"方块像素面积: {ref_px:.1f} px²")

# 7. 比例换算（假设方块真实面积为4cm²）
REF_REAL_AREA = 4.0  
if ref_px > 0:
    scale = REF_REAL_AREA / ref_px
    leaf_real_area = leaf_px * scale
    print(f"比例系数: {scale:.8f} cm²/px")
    print(f"绿叶真实面积: {leaf_real_area:.3f} cm²")
else:
    print("方块面积为0，无法换算！")

# 8. 可视化
vis = img.copy()
if leaf_c is not None: cv2.drawContours(vis, [leaf_c], -1, (0, 0, 255), 2)
if ref_c is not None:  cv2.drawContours(vis, [cv2.convexHull(ref_c)], -1, (0, 255, 0), 2)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("原图(已去噪)"); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(green_mask, cmap='gray'); plt.title("绿色叶片掩膜"); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(blue_mask, cmap='gray'); plt.title("蓝色方块掩膜"); plt.axis('off')
plt.tight_layout(); plt.show()