import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/nsl_kdd.csv', header=None)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df[41])  # 第42列为标签列
label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

for k, v in sorted(label_mapping.items()):
    print(f"{k}: {v}")
