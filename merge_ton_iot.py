import os
import pandas as pd

# 设置你本地的 ToN-IoT 目录路径（修改为你的绝对路径）
data_dir = "D:\python-learn\keras\Train_Test_datasets\Train_Test_Linux_dataset"

# 获取所有 CSV 文件路径
csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

merged_df = pd.DataFrame()

# 遍历每个文件
for file in csv_files:
    try:
        df = pd.read_csv(file)
        df["SourceFile"] = os.path.basename(file)  # 添加来源列
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        print(f"✔ Loaded: {file} ({df.shape[0]} rows)")
    except Exception as e:
        print(f"⚠ Failed to load {file}: {e}")

# 保存合并后的结果
output_path = "ton_iot_one.csv"
merged_df.to_csv(output_path, index=False)
print(f"\n✅ 合并完成，共 {merged_df.shape[0]} 条记录，保存至: {output_path}")
