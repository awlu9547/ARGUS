import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 准备数据
auc_data = {
    'Method': ['ml+graph'] * 5 + ['sml'] * 5 + ['ml'] * 5,
    'AUC': [0.894, 0.873, 0.893, 0.869, 0.866,
            0.867, 0.871, 0.844, 0.849, 0.871,
            0.851, 0.867, 0.840, 0.842, 0.851]
}

# 转换成DataFrame
df = pd.DataFrame(auc_data)

# 设置图形风格
sns.set(style="whitegrid")

# 创建箱线图
plt.figure(figsize=(8, 6))
ax = sns.boxplot(x='Method', y='AUC', data=df, palette="Set2", width=0.5, showmeans=True, meanprops={"marker": "o",
                                                                                                     "markerfacecolor": "red",
                                                                                                     "markeredgecolor": "black"})

# 添加散点（可选，显示每个AUC点）
sns.stripplot(x='Method', y='AUC', data=df, color='black', size=8, jitter=True)

# 设置标题和标签
plt.title('AUC Boxplot for Different Methods')
plt.xlabel('Method')
plt.ylabel('AUC')

# 显示图形
plt.show()
