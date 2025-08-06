from preprocess.preprocess_data import load_and_preprocess
from model.iotfecnn import build_iotfecnn
from model.extract_features import extract_cnn_features
from model.train_rf import train_and_evaluate_rf
import numpy as np
import pandas as pd
import argparse
from preprocess.preprocess_data import load_and_preprocess
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nsl_kdd', help='nsl_kdd or ton_iot')
args = parser.parse_args()
# 加载并预处理数据
X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(dataset=args.dataset)


# === Step 1: 使用 BMECapSA 特征选择（从原始特征中选择）===
from feature_selection.bme_capsa import BME_CapSA

# 将 3D 输入还原为 2D 特征表
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

print(f"使用 BMECapSA 进行特征选择... 原始特征维度: {X_train_2d.shape[1]}")
selector = BME_CapSA(X_train_2d, y_train, pop_size=6, max_iter=10, feature_names=feature_names)

best_subset = selector.optimize()

selected_cols = [i for i, v in enumerate(best_subset) if v == 1]
print(f"选择的特征索引: {selected_cols}")
print(f"被选择的特征名: {[feature_names[i] for i in selected_cols]}")
with open("selected_features.txt", "w") as f:
    for feat in [feature_names[i] for i in selected_cols]:
        f.write(feat + "\n")

X_train_sel = X_train_2d[:, selected_cols]
X_test_sel = X_test_2d[:, selected_cols]

# 转换为 CNN 输入形状
X_train = X_train_sel.reshape(X_train_sel.shape[0], X_train_sel.shape[1], 1)
X_test = X_test_sel.reshape(X_test_sel.shape[0], X_test_sel.shape[1], 1)

# === Step 2: CNN 特征提取模型训练 ===
num_classes = len(np.unique(y_train))  # 自动识别标签类别数
model = build_iotfecnn(input_shape=(X_train.shape[1], 1), num_classes=num_classes)

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# 特征提取
feature_model = extract_cnn_features(model)
cnn_feat_train = feature_model.predict(X_train)
cnn_feat_test = feature_model.predict(X_test)
# 保存中间特征向量
np.save(f"cnn_features_train_{args.dataset}.npy", cnn_feat_train)
np.save(f"cnn_features_test_{args.dataset}.npy", cnn_feat_test)
# 拼接原始 + CNN 特征
X_train_raw = X_train.reshape(X_train.shape[0], -1)
X_test_raw = X_test.reshape(X_test.shape[0], -1)
X_train_all = np.concatenate([X_train_raw, cnn_feat_train], axis=1)
X_test_all = np.concatenate([X_test_raw, cnn_feat_test], axis=1)

# === Step 3: 最终 RandomForest 分类 ===
train_and_evaluate_rf(X_train_all, y_train, X_test_all, y_test)
