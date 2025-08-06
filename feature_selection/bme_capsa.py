# === bme_capsa.py 初始内容（伪代码结构）===
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 适应度函数：特征子集在 RF 上的负准确率 + 特征数
# 用于 BMECapSA 优化搜索
class BME_CapSA:
    def __init__(self, X, y, pop_size=20, max_iter=30, min_features=12, feature_names=None):
        self.X = X
        self.y = y
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.n_features = X.shape[1]
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features)]

    def evaluate(self, bitstring):
        cols = [i for i in range(self.n_features) if bitstring[i] == 1]
        if not cols:
            return 1.0  # 罚分
        clf = RandomForestClassifier(n_estimators=50)
        acc = cross_val_score(clf, self.X[:, cols], self.y, cv=3).mean()
        return 1 - acc + 0.01 * len(cols) / self.n_features

    def optimize(self):
        population = np.random.randint(0, 2, size=(self.pop_size, self.n_features))
        best_score = float('inf')
        best_solution = None

        for gen in range(self.max_iter):
            scores = [self.evaluate(ind) for ind in population]
            best_idx = np.argmin(scores)
            if scores[best_idx] < best_score:
                best_score = scores[best_idx]
                best_solution = population[best_idx]
            population = (population + np.random.randint(0, 2, population.shape)) % 2

        selected_indices = [i for i in range(self.n_features) if best_solution[i] == 1]
        selected_names = [self.feature_names[i] for i in selected_indices]
        print(f"🎯 被选中特征名: {selected_names}")
        return best_solution
