import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def calculate_tamura_coefficient(image):
    # 确保输入图像是灰度图像
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算图像的梯度（使用Sobel算子）
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # x方向的梯度
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # y方向的梯度
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)  # 计算梯度幅度

    # 计算梯度幅度的平均值和标准差
    mean_grad = np.mean(grad_magnitude)  # 平均梯度幅度
    std_grad = np.std(grad_magnitude)    # 梯度幅度的标准差

    # 计算Tamura系数
    tamura_coefficient = np.sqrt(std_grad / mean_grad)

    return tamura_coefficient

# 输入路径（可以是文件或文件夹）
input_path = '/Volumes/U1/SLM/reslt_1010/measure1.jpg'  # 替换为你的输入路径

# 判断输入的是文件还是文件夹
if os.path.isdir(input_path):  # 输入是文件夹
    tc_values = []
    for filename in os.listdir(input_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理JPG和PNG文件
            image_path = os.path.join(input_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                tc = calculate_tamura_coefficient(image)
                tc_values.append(tc)
    
    # 打印所有图像的Tamura系数
    for idx, tc in enumerate(tc_values):
        print(f"Image {idx + 1} Tamura Coefficient: {tc}")
    
    # 绘制Tamura系数的曲线图
    plt.plot(tc_values, marker='o', linestyle='-', color='b')
    plt.title('Tamura Coefficients for Images in Folder')
    plt.xlabel('Image Index')
    plt.ylabel('Tamura Coefficient')
    plt.grid(True)
    plt.show()

elif os.path.isfile(input_path):  # 输入是单个文件
    image = cv2.imread(input_path)
    if image is not None:
        tc = calculate_tamura_coefficient(image)
        print(f"Tamura Coefficient for the input image: {tc}")
    else:
        print("Failed to read the image.")

else:
    print("The specified path is neither a file nor a folder.")
