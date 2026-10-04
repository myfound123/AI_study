import cv2
import matplotlib.pyplot as plt  # 新增这一行

image = cv2.imread('images/leaf.jpg')  

if image is None:
    print("错误：未找到照片！检查路径和文件名")
else:
    print("照片读取成功")
    print("照片的高度是：", image.shape[0])
    print("照片的宽度是：", image.shape[1])

    # 使用 matplotlib 显示图片
    # 注意：cv2读入的是BGR格式，matplotlib显示需要RGB，所以转换一下
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.title("My Leaf")
    plt.axis('off')  # 关掉坐标轴
    plt.show()