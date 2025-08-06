import pandas as pd

# 修改为你的实际路径
csv_path = "D:\python-learn\keras\data/nsl_kdd.csv"

# 加载 CSV 文件
df = pd.read_csv(csv_path)

# 显示 label 分布
if 'label' in df.columns:
    print("✅ 标签分布（label 字段）:")
    print(df['label'].value_counts())
else:
    print("❌ 没有找到 'label' 字段，请确认列名是否为小写/大写一致（如 Label 或 label）")
