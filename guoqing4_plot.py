# 图表导出和画图
real_areas = [4.0,9.0,15.8]
pred_areas = [4.00,8.79,15.39]
import os 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文补丁
plt.rcParams['axes.unicode_minus'] = False   


# 1. 创建 DataFrame（数据表）

df = pd.DataFrame({
    '真实面积(cm²)': real_areas,
    '预测面积(cm²)': pred_areas
})

target_folder = r"C:\Users\xjfou\Desktop\AI_study"

# 2. 导出 Excel（补充绝对路径）
excel_path = os.path.join(target_folder, 'Day4_结果.xlsx')
df.to_excel(excel_path, index=False)
print(f"Excel 已生成！存放位置: {excel_path}")

# 3. 画散点图

plt.scatter(df['真实面积(cm²)'], df['预测面积(cm²)'], color='blue')

# 4. 画一条“完美预测线”（y = x 对角线，作为对照）

plt.plot([0, 20], [0, 20], color='red', linestyle='--', label='完美预测线')

# 5. 图表装饰
plt.xlabel('真实面积 (cm²)')
plt.ylabel('代码预测面积 (cm²)')
plt.title('叶片面积计算器 1.0：预测 vs 真实')
plt.legend() # 显示图例
plt.grid(True) # 显示网格

# 6. 保存图片并显示（未补丁）

plt.savefig('对比散点图.png')
plt.show()

# 昨晚imagej抠出叶片面积为593681px平方，DAy3green跑出580420.5，误差为2.2336%