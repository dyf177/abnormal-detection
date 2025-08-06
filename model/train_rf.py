from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate_rf(X_train_all, y_train, X_test_all, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_all, y_train)
    acc = clf.score(X_test_all, y_test)
    print(f"Random Forest Accuracy: {acc:.4f}")

    # 评估指标输出
    y_pred = clf.predict(X_test_all)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
