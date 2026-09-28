import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============ 工具函数 ============
def extract_mask(img_bgr, lower, upper, extra_mask=None):
    """按HSV阈值提取掩膜，extra_mask用于红色这种需要两段阈值的情况"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lower), np.array(upper))
    if extra_mask is not None:
        m = cv2.bitwise_or(m, extra_mask)
    # 形态学去噪：先开运算去掉小白点，再闭运算补小洞
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

def largest_contour(mask):
    """返回(mask中最大轮廓的像素面积, 轮廓点集)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None
    c = max(contours, key=cv2.contourArea)
    return cv2.contourArea(c), c

# ============ 主流程 ============
img = cv2.imread("leaf_with_ref.jpg")
if img is None:
    raise FileNotFoundError("图片没读到，检查路径/后缀")

# --- 1. 绿色叶片掩膜 ---
leaf_mask = extract_mask(img, [35, 25, 15], [85, 255, 255])

# --- 2. 蓝色方块掩膜（推荐用蓝色，单段阈值简单）---
#     若用红色方块：H分两段(0~10 和 170~180)，见下方注释
ref_mask = extract_mask(img, [100, 150, 50], [130, 255, 255])
# 红色方块版本（备选）：
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# m1 = cv2.inRange(hsv, np.array([0,150,50]),   np.array([10,255,255]))
# m2 = cv2.inRange(hsv, np.array([170,150,50]), np.array([180,255,255]))
# ref_mask = extract_mask(img, [0,150,50], [10,255,255], extra_mask=m2)

# --- 3. 提取最大轮廓的面积 ---
leaf_px, leaf_c = largest_contour(leaf_mask)
ref_px,  ref_c  = largest_contour(ref_mask)

print(f"叶片像素面积 : {leaf_px:.1f} px²")
print(f"方块像素面积 : {ref_px:.1f} px²")

if ref_px == 0:
    raise RuntimeError("没找到参照物！检查蓝色阈值范围，或方块是否被光照过曝")
if leaf_px == 0:
    raise RuntimeError("没找到叶片！检查绿色阈值范围")

# --- 4. 换算真实面积 ---
REF_REAL_AREA = 4.0            # 2cm × 2.05cm = 4.1cm²
scale          = REF_REAL_AREA / ref_px     # 单位：cm²/px
leaf_real_area = leaf_px * scale

print(f"比例系数     : {scale:.8f} cm²/px")
print(f"叶片真实面积 : {leaf_real_area:.3f} cm²")

# --- 5. 可视化验证 ---
vis = img.copy()
cv2.drawContours(vis, [leaf_c], -1, (0, 0, 255), 2)   # 叶片红色描边
cv2.drawContours(vis, [ref_c],  -1, (0, 255, 0), 2)   # 方块绿色描边

plt.figure(figsize=(14, 4))
for i, (im, t) in enumerate(zip(
        [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), leaf_mask, ref_mask, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)],
        ["原图", "叶片掩膜", "参照物掩膜", "双轮廓叠加"])):
    plt.subplot(1, 4, i+1); plt.imshow(im, cmap='gray' if i in (1,2) else None)
    plt.title(t); plt.axis('off')
plt.tight_layout(); plt.show()
