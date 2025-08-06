构建一个可模块化切换数据集的异常流量检测系统，以 BMECapSA + CNN + Random Forest 为核心，实现多模型融合、特征优化、自适应分类，支持 NetFlow/PCAP 衍生字段分析。
# 🚀 Anomaly Detection System (NSL-KDD & ToN-IoT)

This repository implements a modular, feature-optimized, and ensemble-based anomaly detection system using **BMECapSA**, **1D CNN**, and **Random Forest** classifiers. The system supports two datasets: **NSL-KDD** and **ToN-IoT**, with automatic feature selection and multi-model fusion.

---

## ✨ Features

* ✅ Support for **NetFlow-style features** from NSL-KDD and **sensor/IoT fields** from ToN-IoT
* ✅ Plug-and-play pipeline via `main_pipeline.py`
* ✅ **BMECapSA** (Binary Multi-Strategy Ensemble Capuchin Search Algorithm) for feature subset optimization
* ✅ **1D CNN** for deep behavioral feature extraction
* ✅ **Random Forest** for ensemble fusion & evaluation
* ✅ Classification report, accuracy metrics, and optional confusion matrix

---

## 📂 Project Structure

```
keras/
├── main_pipeline.py           # 主流程入口，控制训练与评估
├── data/
│   ├── nsl_kdd.csv
│   └── ton_iot.csv
├── preprocess/
│   └── preprocess_data.py     # 数据预处理，含标签清洗、归一化、泄露字段提示等
├── feature_selection/
│   └── bme_capsa.py           # BMECapSA 特征优化算法（待定）
├── model/
│   ├── iotfecnn.py           # CNN 模型训练
│   ├── extract_features.py    # ✅ 提取 CNN 中间特征向量的模块
│   └── train_rf.py            # 随机森林训练与评估 # 构建并训练 Random Forest 模型，#混淆矩阵、分类报告、可视化等工具函数
```

---

## 🚀 Quick Start

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

* `data/nsl_kdd.csv`
* `data/ton_iot_all.csv`

### 3. Run pipeline

```bash
# For NSL-KDD
python main_pipeline.py --dataset nsl_kdd

# For ToN-IoT
python main_pipeline.py --dataset ton_iot
```

---

## 📈 Pipeline Overview

1. **Load & Preprocess**

   * Label encoding, NaN removal, MinMax normalization
   * Optional detection of label leakage (e.g., `type`, `date`, `temp_condition`)

2. **Feature Selection via BMECapSA**

   * Randomly initializes feature subsets
   * Evaluates subsets with RandomForest
   * Evolves toward best-performing subset

3. **CNN Model Training**

   * Uses 1D ConvNet to capture temporal/dependency features
   * Accuracy and loss reported per epoch

4. **Random Forest Training & Evaluation**

   * Trains on selected features
   * Outputs accuracy and detailed classification report

---

## 🌐 Example Output (ToN-IoT)

```
📅 Label Distribution:
Train: 1: 124859, 0: 84036
Test : 1: 31260,  0: 20964

BMECapSA selected 17 features
CNN accuracy: 1.0000
Random Forest Accuracy: 0.9914

=== Classification Report ===
              precision    recall  f1-score   support
           0     0.9921     0.9856    0.9888     20964
           1     0.9903     0.9967    0.9935     31260
```

---

## 🛠️ Configuration Tips

* To **limit feature subset size**, modify `min_features` and `max_features` in `bme_capsa.py`
* To **remove label-leaking fields**, add column names like `type`, `date`, `time` to drop list in `preprocess_data.py`
* To **visualize confusion matrix**, use matplotlib + seaborn in `utils/train_rf.py`

---

## 🚜 Next Steps

* [ ] Add support for AutoEncoder / PCA compression
* [ ] Add XGBoost / LightGBM to ensemble
* [ ] Export trained model (`.h5`, `.pkl`) for deployment
* [ ] Flask / FastAPI wrapper for real-time detection API

---

## 🎓 Citation / Acknowledgments

* BMECapSA algorithm adapted from related IDS research
* NSL-KDD and ToN-IoT datasets are publicly available for academic use
  The theoretical support from 《Anomaly-based-intrusion-detection-system-in-the-Interne_2023_Journal-of-Para》 
---

## 📁 Dataset Notes

| Dataset | Description                     | File              |
| ------- | ------------------------------- | ----------------- |
| NSL-KDD | 1999-style NetFlow anomaly data | `nsl_kdd.csv`     |
| ToN-IoT | Realistic IoT smart home logs   | `ton_iot_all.csv` |

---

## 📢 Feedback

Issues and improvements welcome. Reach out via GitHub Issues.

