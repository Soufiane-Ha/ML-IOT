# ML-IOT

🔥 AI for IoT Practical 1: Fire Alarm Detection
Binary Classification using Logistic Regression and XGBoost
Objective
The objective of this practical work is to build a machine learning classification system capable of predicting whether an IoT smoke detector should trigger a fire alarm based on environmental sensor readings.

Models
Logistic Regression — Core Task
XGBoost — Bonus
Target
1 → Fire/Smoke detected
0 → Normal conditions



# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import glob
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

print("Libraries imported successfully.")

# ============================================================
# 2. FIND DATASET FILES
# ============================================================

csv_files = glob.glob(
    "/kaggle/input/**/*.csv",
    recursive=True
)
print("CSV files found:")
for file in csv_files:
    print(file)

# ============================================================
# 3. LOAD DATASET
# ============================================================

if len(csv_files) == 0:
    raise FileNotFoundError(
        "No CSV file found. Please add the Smoke Detection Dataset to Kaggle."
    )

# Use the first CSV file found
file_path = csv_files[0]

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("File:", file_path)
print("Shape:", df.shape)

# ============================================================
# 4. FIRST LOOK AT THE DATA
# ============================================================

print("First 5 rows:")
display(df.head())

# ============================================================
# 4. FIRST LOOK AT THE DATA
# ============================================================

print("First 5 rows:")
display(df.head())

print("\nStatistical description:")
display(df.describe())

# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

missing_values = df.isnull().sum()

print("Missing values per column:")
display(missing_values)

print("\nTotal missing values:", missing_values.sum())

# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

print("Missing values after preprocessing:")
print(df.isnull().sum().sum())

# ============================================================
# 7. DISPLAY COLUMNS
# ============================================================

print("Dataset columns:")

for i, column in enumerate(df.columns):
    print(i, ":", column)

# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

target_column = "Fire Alarm"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' was not found. "
        f"Available columns: {list(df.columns)}"
    )

print("Target distribution:")
display(df[target_column].value_counts())

print("\nTarget percentages:")
display(df[target_column].value_counts(normalize=True) * 100)

# ============================================================
# 9. VISUALIZE TARGET DISTRIBUTION
# ============================================================

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x=target_column
)

plt.title("Fire Alarm Class Distribution")
plt.xlabel("Fire Alarm")
plt.ylabel("Number of Samples")

plt.show()

# ============================================================
# 10. DEFINE FEATURES (X) AND TARGET (y)
# ============================================================

# Remove target from input features
X = df.drop(columns=[target_column])

# Remove non-numeric/time identifier columns
X = X.select_dtypes(include=np.number)

# Target
y = df[target_column]

print("Features (X):")
print(X.columns.tolist())

print("\nTarget (y):")
print(y.name)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# 11. CHECK X AND y
# ============================================================

print("X:")
display(X.head())

print("\ny:")
display(y.head())

# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining percentage:",
      round(len(X_train) / len(X) * 100, 2), "%")

print("Testing percentage:",
      round(len(X_test) / len(X) * 100, 2), "%")

# ============================================================
# 13. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed successfully.")

print("\nTraining data shape:", X_train_scaled.shape)
print("Testing data shape:", X_test_scaled.shape)

# ============================================================
# 14. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model
logistic_model.fit(
    X_train_scaled,
    y_train
)

print("Logistic Regression model trained successfully.")

# ============================================================
# 15. MAKE PREDICTIONS
# ============================================================

y_pred_logistic = logistic_model.predict(X_test_scaled)

print("Predictions generated successfully.")

print("\nFirst 20 predictions:")
print(y_pred_logistic[:20])

# ============================================================
# 16. EVALUATE LOGISTIC REGRESSION
# ============================================================

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

logistic_precision = precision_score(
    y_test,
    y_pred_logistic,
)

logistic_recall = recall_score(
    y_test,
    y_pred_logistic,
)

logistic_f1 = f1_score(
    y_test,
    y_pred_logistic,
)

print("===== Logistic Regression Results =====")

print(f"Accuracy :  {logistic_accuracy:.4f}")
print(f"Precision:  {logistic_precision:.4f}")
print(f"Recall   :  {logistic_recall:.4f}")
print(f"F1-Score :  {logistic_f1:.4f}")

# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_logistic,
        target_names=["Normal", "Fire"],
    )
)

# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm_logistic = confusion_matrix(
    y_test,
    y_pred_logistic
)

print("Confusion Matrix:")
print(cm_logistic)

# ============================================================
# 19. VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fire"],
    yticklabels=["Normal", "Fire"]
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()

# ============================================================
# 20. EXTRACT TN, FP, FN, TP
# ============================================================

tn, fp, fn, tp = cm_logistic.ravel()

print("True Negative (TN):", tn)
print("False Positive (FP):", fp)
print("False Negative (FN):", fn)
print("True Positive (TP):", tp)

# ============================================================
# 21. FIRE DETECTION SAFETY ANALYSIS
# ============================================================

print("Fire Detection Analysis")
print("------------------------")

print("False Negatives:", fn)
print("False Positives:", fp)

print(
    "\nFalse Negative means that the model predicted NORMAL "
    "while the actual condition was FIRE."
)

print(
    "In a fire alarm system, False Negatives are particularly dangerous "
    "because a real fire may not trigger an alarm."
)

#XGBoost

# ============================================================
# 22. XGBOOST IMPORT
# ============================================================

print("XGBoost imported successfully.")

# ============================================================
# 23. XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

# Train XGBoost
xgb_model.fit(
    X_train_scaled,
    y_train
)

print("XGBoost model trained successfully.")

# ============================================================
# 24. XGBOOST PREDICTIONS
# ============================================================

y_pred_xgb = xgb_model.predict(X_test_scaled)

print("XGBoost predictions generated successfully.")

print("\nFirst 20 predictions:")
print(y_pred_xgb[:20])

# ============================================================
# 25. EVALUATE XGBOOST
# ============================================================

xgb_accuracy = accuracy_score(
    y_test,
    y_pred_xgb
)

xgb_precision = precision_score(
    y_test,
    y_pred_xgb,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    y_pred_xgb,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    y_pred_xgb,
    zero_division=0
)

print("========== XGBoost Results ==========")

print(f"Accuracy :  {xgb_accuracy:.4f}")
print(f"Precision:  {xgb_precision:.4f}")
print(f"Recall   :  {xgb_recall:.4f}")
print(f"F1-Score :  {xgb_f1:.4f}")

# ============================================================
# 26. XGBOOST CLASSIFICATION REPORT
# ============================================================

print("XGBoost Classification Report:")

print(
    classification_report(
        y_test,
        y_pred_xgb,
        target_names=["Normal", "Fire"],
        zero_division=0
    )
)

# ============================================================
# 27. XGBOOST CONFUSION MATRIX
# ============================================================

cm_xgb = confusion_matrix(
    y_test,
    y_pred_xgb
)

print("XGBoost Confusion Matrix:")
print(cm_xgb)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_xgb,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Normal", "Fire"],
    yticklabels=["Normal", "Fire"]
)

plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()

# ============================================================
# 28. XGBOOST TP / TN / FP / FN
# ============================================================

tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()

print("True Negative (TN):", tn_xgb)
print("False Positive (FP):", fp_xgb)
print("False Negative (FN):", fn_xgb)
print("True Positive (TP):", tp_xgb)

# ============================================================
# 29. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    
    "Logistic Regression": [
        logistic_accuracy,
        logistic_precision,
        logistic_recall,
        logistic_f1
    ],
    
    "XGBoost": [
        xgb_accuracy,
        xgb_precision,
        xgb_recall,
        xgb_f1
    ]
})

display(comparison)

# ============================================================
# 30. VISUAL MODEL COMPARISON
# ============================================================

comparison_plot = comparison.set_index("Metric")

comparison_plot.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Logistic Regression vs XGBoost")
plt.ylabel("Score")
plt.ylim(0, 1.05)

plt.xticks(rotation=0)

plt.legend(
    title="Model"
)

plt.show()

# ============================================================
# 31. RECALL COMPARISON
# ============================================================

print("========== Recall Comparison ==========")

print(
    f"Logistic Regression Recall: {logistic_recall:.4f}"
)

print(
    f"XGBoost Recall: {xgb_recall:.4f}"
)

if xgb_recall > logistic_recall:
    print("\nXGBoost has the higher Recall.")
    
elif logistic_recall > xgb_recall:
    print("\nLogistic Regression has the higher Recall.")
    
else:
    print("\nBoth models have the same Recall.")

# ============================================================
# 32. SAFETY ANALYSIS
# ============================================================

print("========== Fire Detection Safety Analysis ==========")

if fn_xgb < fn:
    print(
        "XGBoost produced fewer False Negatives than Logistic Regression."
    )
    
elif fn < fn_xgb:
    print(
        "Logistic Regression produced fewer False Negatives than XGBoost."
    )
    
else:
    print(
        "Both models produced the same number of False Negatives."
    )

print(
    "\nWhy are False Negatives important?"
)

print(
    "A False Negative occurs when the actual condition is FIRE "
    "but the model predicts NORMAL."
)

print(
    "For an IoT fire alarm system, reducing False Negatives is important "
    "because failing to detect a real fire can have serious consequences."
)

# ============================================================
# 33. FINAL RESULTS
# ============================================================

final_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "XGBoost"
    ],
    
    "Accuracy": [
        logistic_accuracy,
        xgb_accuracy
    ],
    
    "Precision": [
        logistic_precision,
        xgb_precision
    ],
    
    "Recall": [
        logistic_recall,
        xgb_recall
    ],
    
    "F1-Score": [
        logistic_f1,
        xgb_f1
    ]
})

display(final_results)



Libraries imported successfully.
CSV files found:
/kaggle/input/datasets/deepcontractor/smoke-detection-dataset/smoke_detection_iot.csv
Dataset loaded successfully!
File: /kaggle/input/datasets/deepcontractor/smoke-detection-dataset/smoke_detection_iot.csv
Shape: (62630, 16)
First 5 rows:
Unnamed: 0	UTC	Temperature[C]	Humidity[%]	TVOC[ppb]	eCO2[ppm]	Raw H2	Raw Ethanol	Pressure[hPa]	PM1.0	PM2.5	NC0.5	NC1.0	NC2.5	CNT	Fire Alarm
0	0	1654733331	20.000	57.36	0	400	12306	18520	939.735	0.0	0.0	0.0	0.0	0.0	0	0
1	1	1654733332	20.015	56.67	0	400	12345	18651	939.744	0.0	0.0	0.0	0.0	0.0	1	0
2	2	1654733333	20.029	55.96	0	400	12374	18764	939.738	0.0	0.0	0.0	0.0	0.0	2	0
3	3	1654733334	20.044	55.28	0	400	12390	18849	939.736	0.0	0.0	0.0	0.0	0.0	3	0
4	4	1654733335	20.059	54.69	0	400	12403	18921	939.744	0.0	0.0	0.0	0.0	0.0	4	0
First 5 rows:
Unnamed: 0	UTC	Temperature[C]	Humidity[%]	TVOC[ppb]	eCO2[ppm]	Raw H2	Raw Ethanol	Pressure[hPa]	PM1.0	PM2.5	NC0.5	NC1.0	NC2.5	CNT	Fire Alarm
0	0	1654733331	20.000	57.36	0	400	12306	18520	939.735	0.0	0.0	0.0	0.0	0.0	0	0
1	1	1654733332	20.015	56.67	0	400	12345	18651	939.744	0.0	0.0	0.0	0.0	0.0	1	0
2	2	1654733333	20.029	55.96	0	400	12374	18764	939.738	0.0	0.0	0.0	0.0	0.0	2	0
3	3	1654733334	20.044	55.28	0	400	12390	18849	939.736	0.0	0.0	0.0	0.0	0.0	3	0
4	4	1654733335	20.059	54.69	0	400	12403	18921	939.744	0.0	0.0	0.0	0.0	0.0	4	0

Statistical description:
Unnamed: 0	UTC	Temperature[C]	Humidity[%]	TVOC[ppb]	eCO2[ppm]	Raw H2	Raw Ethanol	Pressure[hPa]	PM1.0	PM2.5	NC0.5	NC1.0	NC2.5	CNT	Fire Alarm
count	62630.000000	6.263000e+04	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000	62630.000000
mean	31314.500000	1.654792e+09	15.970424	48.539499	1942.057528	670.021044	12942.453936	19754.257912	938.627649	100.594309	184.467770	491.463608	203.586487	80.049042	10511.386157	0.714626
std	18079.868017	1.100025e+05	14.359576	8.865367	7811.589055	1905.885439	272.464305	609.513156	1.331344	922.524245	1976.305615	4265.661251	2214.738556	1083.383189	7597.870997	0.451596
min	0.000000	1.654712e+09	-22.010000	10.740000	0.000000	400.000000	10668.000000	15317.000000	930.852000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	15657.250000	1.654743e+09	10.994250	47.530000	130.000000	400.000000	12830.000000	19435.000000	938.700000	1.280000	1.340000	8.820000	1.384000	0.033000	3625.250000	0.000000
50%	31314.500000	1.654762e+09	20.130000	50.150000	981.000000	400.000000	12924.000000	19501.000000	938.816000	1.810000	1.880000	12.450000	1.943000	0.044000	9336.000000	1.000000
75%	46971.750000	1.654778e+09	25.409500	53.240000	1189.000000	438.000000	13109.000000	20078.000000	939.418000	2.090000	2.180000	14.420000	2.249000	0.051000	17164.750000	1.000000
max	62629.000000	1.655130e+09	59.930000	75.200000	60000.000000	60000.000000	13803.000000	21410.000000	939.861000	14333.690000	45432.260000	61482.030000	51914.680000	30026.438000	24993.000000	1.000000
Missing values per column:
Unnamed: 0        0
UTC               0
Temperature[C]    0
Humidity[%]       0
TVOC[ppb]         0
eCO2[ppm]         0
Raw H2            0
Raw Ethanol       0
Pressure[hPa]     0
PM1.0             0
PM2.5             0
NC0.5             0
NC1.0             0
NC2.5             0
CNT               0
Fire Alarm        0
dtype: int64

Total missing values: 0
Missing values after preprocessing:
0
Dataset columns:
0 : Unnamed: 0
1 : UTC
2 : Temperature[C]
3 : Humidity[%]
4 : TVOC[ppb]
5 : eCO2[ppm]
6 : Raw H2
7 : Raw Ethanol
8 : Pressure[hPa]
9 : PM1.0
10 : PM2.5
11 : NC0.5
12 : NC1.0
13 : NC2.5
14 : CNT
15 : Fire Alarm
Target distribution:
Fire Alarm
1    44757
0    17873
Name: count, dtype: int64

Target percentages:
Fire Alarm
1    71.462558
0    28.537442
Name: proportion, dtype: float64

Features (X):
['Unnamed: 0', 'UTC', 'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT']

Target (y):
Fire Alarm

X shape: (62630, 15)
y shape: (62630,)
X:
Unnamed: 0	UTC	Temperature[C]	Humidity[%]	TVOC[ppb]	eCO2[ppm]	Raw H2	Raw Ethanol	Pressure[hPa]	PM1.0	PM2.5	NC0.5	NC1.0	NC2.5	CNT
0	0	1654733331	20.000	57.36	0	400	12306	18520	939.735	0.0	0.0	0.0	0.0	0.0	0
1	1	1654733332	20.015	56.67	0	400	12345	18651	939.744	0.0	0.0	0.0	0.0	0.0	1
2	2	1654733333	20.029	55.96	0	400	12374	18764	939.738	0.0	0.0	0.0	0.0	0.0	2
3	3	1654733334	20.044	55.28	0	400	12390	18849	939.736	0.0	0.0	0.0	0.0	0.0	3
4	4	1654733335	20.059	54.69	0	400	12403	18921	939.744	0.0	0.0	0.0	0.0	0.0	4

y:
0    0
1    0
2    0
3    0
4    0
Name: Fire Alarm, dtype: int64
Training samples: 50104
Testing samples: 12526

Training percentage: 80.0 %
Testing percentage: 20.0 %
Feature scaling completed successfully.

Training data shape: (50104, 15)
Testing data shape: (12526, 15)
Logistic Regression model trained successfully.
Predictions generated successfully.

First 20 predictions:
[1 1 0 0 1 1 0 1 0 0 1 0 0 1 1 1 1 0 0 1]
===== Logistic Regression Results =====
Accuracy :  0.9869
Precision:  0.9923
Recall   :  0.9894
F1-Score :  0.9908
Classification Report:
              precision    recall  f1-score   support

      Normal       0.97      0.98      0.98      3575
        Fire       0.99      0.99      0.99      8951

    accuracy                           0.99     12526
   macro avg       0.98      0.99      0.98     12526
weighted avg       0.99      0.99      0.99     12526

Confusion Matrix:
[[3506   69]
 [  95 8856]]

True Negative (TN): 3506
False Positive (FP): 69
False Negative (FN): 95
True Positive (TP): 8856
Fire Detection Analysis
------------------------
False Negatives: 95
False Positives: 69

False Negative means that the model predicted NORMAL while the actual condition was FIRE.
In a fire alarm system, False Negatives are particularly dangerous because a real fire may not trigger an alarm.
XGBoost imported successfully.
XGBoost model trained successfully.
XGBoost predictions generated successfully.

First 20 predictions:
[1 1 0 0 1 1 0 1 0 1 1 0 0 1 1 1 1 0 0 1]
========== XGBoost Results ==========
Accuracy :  0.9999
Precision:  1.0000
Recall   :  0.9999
F1-Score :  0.9999
XGBoost Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00      3575
        Fire       1.00      1.00      1.00      8951

    accuracy                           1.00     12526
   macro avg       1.00      1.00      1.00     12526
weighted avg       1.00      1.00      1.00     12526

XGBoost Confusion Matrix:
[[3575    0]
 [   1 8950]]

True Negative (TN): 3575
False Positive (FP): 0
False Negative (FN): 1
True Positive (TP): 8950
Metric	Logistic Regression	XGBoost
0	Accuracy	0.986907	0.999920
1	Precision	0.992269	1.000000
2	Recall	0.989387	0.999888
3	F1-Score	0.990826	0.999944

========== Recall Comparison ==========
Logistic Regression Recall: 0.9894
XGBoost Recall: 0.9999

XGBoost has the higher Recall.
========== Fire Detection Safety Analysis ==========
XGBoost produced fewer False Negatives than Logistic Regression.

Why are False Negatives important?
A False Negative occurs when the actual condition is FIRE but the model predicts NORMAL.
For an IoT fire alarm system, reducing False Negatives is important because failing to detect a real fire can have serious consequences.
Model	Accuracy	Precision	Recall	F1-Score
0	Logistic Regression	0.986907	0.992269	0.989387	0.990826
1	XGBoost	0.999920	1.000000	0.999888	0.999944



📝 Conclusion
This practical work implemented a binary classification system for an IoT-based fire and smoke detection application.

First, the dataset was loaded and inspected using Pandas. Missing values were checked and handled when necessary. The data was then divided into input features (X) and the target variable (y), where:

0 represents normal conditions.
1 represents fire/smoke detection.
The dataset was divided into 80% training data and 20% testing data.

For Logistic Regression, StandardScaler was applied to standardize the input features before training. The model was then evaluated using Accuracy, Precision, Recall, and F1-Score.

The Confusion Matrix was also analyzed using True Positives, True Negatives, False Positives, and False Negatives.

Recall is particularly important for this application because a False Negative means that an actual fire condition was classified as normal. Such an error could prevent the IoT fire alarm from generating a warning.

As a bonus, an XGBoost classifier was also implemented and evaluated using the same train/test split. The results of both models were compared in terms of Accuracy, Precision, Recall, and F1-Score.

In a real-world fire detection system, model selection should not rely only on Accuracy. Recall and the number of False Negatives are especially important because detecting as many real fire events as possible is a critical safety requirement.

Overall, this practical demonstrates how Machine Learning can be integrated with IoT sensor data to build an intelligent fire alarm detection system.