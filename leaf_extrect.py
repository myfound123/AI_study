import cv2
import matplotlib.pylab as plt
image = cv2.imread('images/leaf.jpg')
if image is None:
    print("错了，照片呢？")
else:
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_green = (25,40,40)
    upper_green = (90,255,255)
    mask = cv2.inRange(hsv_image,lower_green,upper_green)
    
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 4. 【核心修改】把原图和 Mask 横向拼在一起
    combined = cv2.hconcat([image, mask_3ch])

    # 5. 显示拼接后的对比图
    cv2.imshow("Original vs Mask", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

