import os
import cv2
import numpy as np
import pandas as pd

def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0.0, None
    valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
    if not valid_contours: return 0.0, None
    c = max(valid_contours, key=cv2.contourArea)
    return cv2.contourArea(c), c

# 1. 沿用之前写好的 extract_area() 函数
def extract_area(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0
    
    img = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 提取蓝色方块
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 20]), np.array([140, 255, 255]))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    # 提取绿色叶片
    leaf_mask = cv2.inRange(hsv, np.array([35, 25, 15]), np.array([85, 255, 255]))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
     # 面积提取
    leaf_px, _ = largest_contour(leaf_mask)
    ref_px, ref_c = largest_contour(blue_mask)
    # 凸包修复蓝方块
    if ref_c is not None:
        ref_px = cv2.contourArea(cv2.convexHull(ref_c))
    
    if ref_px > 0:
        scale = 4.0 / ref_px
        leaf_real = leaf_px * scale
        return leaf_real
    else:
        return 0

# 2. 批量遍历
image_folder = "images"
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

data_list = []  # 空列表，装所有行

for file_name in image_files:
    if not file_name.startswith("leaf_"):
        continue
    img_path = os.path.join(image_folder, file_name)
    # 调用函数计算真实面积
    real_area = extract_area(img_path)  # 这里需要你传参数
    
    # 将结果存入字典，再放进列表
    data_list.append({
        "文件名": file_name,
        "预测面积(cm²)": round(real_area, 2)  # 保留两位小数
    })

# 3. 生成 Pandas 表格
df = pd.DataFrame(data_list)

# 4. 导出 Excel (注意主路径的防坑提示)
df.to_excel("Day5_批量结果.xlsx", index=False)

print("Excel 生成完毕！")
