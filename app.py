import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

import matplotlib.pyplot as plt
import openpyxl
from openpyxl import load_workbook

st.title("Titanic Survival Prediction")

# --------------------------------------------------
# Load Titanic Dataset
# --------------------------------------------------

df = sns.load_dataset("titanic")

# Add Serial Number
df.insert(0, "Serial_Number", range(1, len(df)+1))

st.subheader("Raw Dataset")
st.dataframe(df.head())

# --------------------------------------------------
# Target Variable
# --------------------------------------------------

target = "survived"

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------

data = df.copy()

# Drop high missing columns
drop_cols = ['deck', 'embark_town', 'alive', 'class', 'who']

for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

# Separate X and y
X = data.drop(columns=[target])
y = data[target]

# Remove Serial Number from predictors
X = X.drop(columns=["Serial_Number"])

# Handle Missing Values
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Label Encoding
le = LabelEncoder()

for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Create Train/Test Data for Excel
# --------------------------------------------------

train_df = data.loc[X_train.index].copy()
test_df = data.loc[X_test.index].copy()

# --------------------------------------------------
# Model Building
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

fpr_value = FP / (FP + TN)

tpr_value = TP / (TP + FN)

roc_auc = roc_auc_score(y_test, y_prob)

# --------------------------------------------------
# ROC Curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "Threshold": thresholds
})

# --------------------------------------------------
# Display Metrics
# --------------------------------------------------

st.subheader("Model Evaluation")

metrics_df = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "FPR",
        "TPR",
        "ROC AUC"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        fpr_value,
        tpr_value,
        roc_auc
    ]
})

st.dataframe(metrics_df)

# --------------------------------------------------
# Display Confusion Matrix
# --------------------------------------------------

cm_df = pd.DataFrame(
    cm,
    columns=["Predicted No", "Predicted Yes"],
    index=["Actual No", "Actual Yes"]
)

st.subheader("Confusion Matrix")

st.dataframe(cm_df)

# --------------------------------------------------
# ROC Curve Plot
# --------------------------------------------------

st.subheader("ROC Curve")

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

ax.plot([0,1],[0,1],'--')

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title("ROC Curve")

ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Export to Excel
# --------------------------------------------------

file_name = "Titanic_Model_Output.xlsx"

with pd.ExcelWriter(
    file_name,
    engine='openpyxl'
) as writer:

    # Sheet 1
    df.to_excel(
        writer,
        sheet_name="Raw Data",
        index=False
    )

    # Sheet 2
    train_df.to_excel(
        writer,
        sheet_name="Train Data",
        index=False
    )

    # Sheet 3
    test_df.to_excel(
        writer,
        sheet_name="Test Data",
        index=False
    )

    # Sheet 4
    cm_df.to_excel(
        writer,
        sheet_name="Confusion Matrix"
    )

    # Sheet 5
    metrics_df.to_excel(
        writer,
        sheet_name="Model Evaluation",
        index=False
    )

    # Sheet 6
    roc_df.to_excel(
        writer,
        sheet_name="ROC Curve Data",
        index=False
    )

st.success(
    f"Excel file generated successfully: {file_name}"
)

# Download Button
with open(file_name, "rb") as file:

    st.download_button(
        label="Download Excel File",
        data=file,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )