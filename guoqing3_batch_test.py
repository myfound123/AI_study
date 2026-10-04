import os  # 👈 这是今天要学的新模块，专门用来处理文件路径
import cv2
import numpy as np

# 定义已知参数
# 假设我们之前的比例尺是 0.00003161
RATIO = 0.00003161

# 2. 定义一个简化版的图像处理函数（模拟你之前的提取逻辑）
def extract_area(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0
    
    img = cv2.medianBlur(img, 5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 提取蓝色方块（用作参照物，保证比例尺有效性）
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 20]), np.array([140, 255, 255]))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    # 提取绿色叶片
    leaf_mask = cv2.inRange(hsv, np.array([35, 25, 15]), np.array([85, 255, 255]))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    
    # 找轮廓取面积（这里假设你已经封装好了largest_contour，直接调用）
    # ... 此处省略具体轮廓计算，我们假定它返回了叶片像素面积
    leaf_px = 580420.5 # 假装这里算出了叶片的像素面积
    return leaf_px

# ================= 主程序 =================

# 3. 设定图片文件夹路径
image_folder = "images" 

# 4. 获取文件夹下所有的图片文件
# 提示：用 os.listdir() 获取文件夹下所有文件，并用列表推导式过滤出 .jpg 结尾的文件
# 语法提示：[变量名 for 变量名 in os.listdir(路径) if 条件]
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 5. 准备一个空列表，用来存放结果
results = []

# 6. 遍历所有图片，进行计算
# 提示：使用 for 循环
for file_name in image_files:
    # 6.1 拼接完整的图片路径
    # 提示：使用 os.path.join(文件夹路径, 文件名)
    img_path = os.path.join(image_folder, file_name)
    
    # 6.2 调用函数计算像素面积
    px_area = extract_area(img_path)
    
    # 6.3 像素面积换算为真实面积
    real_area = px_area * RATIO
    
    # 6.4 将结果存入列表
    # 提示：使用 list.append() 方法，存入一个元组或字典
if px_area > 0:
    results.append({"file": file_name, "area": real_area})
    
    print(f"处理完毕：{file_name}，预测面积：{real_area:.2f} cm²")
else:
    print(f"【警告】{file_name} 读取失败或未找到有效轮廓，已跳过计算。")
# 7. 打印最终结果
print("\n=== 最终结果列表 ===")
for res in results:
    print(res)
