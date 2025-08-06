import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
def load_and_preprocess(csv_path, label_col='label'):
    df = pd.read_csv(csv_path)
def load_and_preprocess(csv_path=None, dataset='nsl_kdd'):
    if dataset == 'nsl_kdd':
        csv_path = csv_path or 'data/nsl_kdd.csv'
        column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
        ]
        df = pd.read_csv(csv_path, names=column_names, header=None, low_memory=False)
        label_col = 'label'
    elif dataset == 'ton_iot':
        csv_path = csv_path or 'data/ton_iot.csv'
        df = pd.read_csv(csv_path, low_memory=False)
        if 'Label' in df.columns:
            label_col = 'Label'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            raise ValueError("未找到 label 或 Label 列！")
    else:
        raise ValueError("Unknown dataset. Use 'nsl_kdd' or 'ton_iot'")

    print(f"数据形状: {df.shape}")
    print("前几列：", df.columns[:5])
    print(df.head(2))
    # 检查潜在字段是否与标签强相关（例如 type/temp_condition）
    suspicious_cols = ['type', 'temp_condition', 'device', 'attack', 'date', 'time']
    print("\n🔍 可疑字段与标签关系（Top 3 类）:")
    for col in suspicious_cols:
        if col in df.columns:
            print(f"\n【{col}】列:")
            print(df.groupby(col)[label_col].value_counts().head(3))

    df = df.dropna(subset=[label_col])
    leaky_cols = ['type', 'temp_condition', 'date', 'time']
    df = df.drop(columns=[c for c in leaky_cols if c in df.columns])

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # ✅ 标签分布检查
    print("\n✅ 标签分布：")
    print("训练集：")
    print(pd.Series(y_train).value_counts())
    print("测试集：")
    print(pd.Series(y_test).value_counts())
    feature_names = X.columns.tolist()
    return X_train, X_test, y_train, y_test, X.columns.tolist()

