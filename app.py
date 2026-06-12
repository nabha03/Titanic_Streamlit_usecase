import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
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

from io import BytesIO

st.set_page_config(page_title="Titanic Survival Prediction")

st.title("Titanic Survival Prediction")

# ==========================
# Load Dataset
# ==========================

df = sns.load_dataset('titanic')

# Serial Number
df.insert(0, "Serial_Number", range(1, len(df)+1))

raw_df = df.copy()

st.subheader("Raw Data")
st.dataframe(df.head())

# ==========================
# Data Preprocessing
# ==========================

target = "survived"

# Drop high-cardinality columns
drop_cols = ['alive', 'who', 'adult_male', 'deck', 'embark_town', 'class']

df = df.drop(columns=drop_cols)

# Separate Target
y = df[target]

# Predictor Set
X = df.drop(columns=[target])

# Save Serial Number separately
serial_numbers = X['Serial_Number']

# Remove Serial Number from training
X = X.drop(columns=['Serial_Number'])

# Encode categorical columns
le_dict = {}

for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Missing Value Imputation
num_cols = X.select_dtypes(include=np.number).columns

imputer = SimpleImputer(strategy='median')
X[num_cols] = imputer.fit_transform(X[num_cols])

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test, sn_train, sn_test = train_test_split(
    X,
    y,
    serial_numbers,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Model Training
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_prob = model.predict_proba(X_train)[:,1]
test_prob = model.predict_proba(X_test)[:,1]

# ==========================
# Metrics
# ==========================

cm = confusion_matrix(y_test, test_pred)

tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)

fpr_metric = fp/(fp+tn)
tpr_metric = tp/(tp+fn)

roc_auc = roc_auc_score(y_test, test_prob)

fpr, tpr, thresholds = roc_curve(y_test, test_prob)

# ==========================
# Display Metrics
# ==========================

st.subheader("Model Evaluation")

st.write("Accuracy :", round(accuracy,4))
st.write("Precision :", round(precision,4))
st.write("Recall :", round(recall,4))
st.write("F1 Score :", round(f1,4))
st.write("FPR :", round(fpr_metric,4))
st.write("TPR :", round(tpr_metric,4))
st.write("ROC-AUC :", round(roc_auc,4))

# ==========================
# Create Train Sheet
# ==========================

train_df = X_train.copy()

train_df.insert(0, "Serial_Number", sn_train.values)

train_df["Actual"] = y_train.values
train_df["Predicted"] = train_pred
train_df["Probability"] = train_prob

# ==========================
# Create Test Sheet
# ==========================

test_df = X_test.copy()

test_df.insert(0, "Serial_Number", sn_test.values)

test_df["Actual"] = y_test.values
test_df["Predicted"] = test_pred
test_df["Probability"] = test_prob

# ==========================
# Confusion Matrix Sheet
# ==========================

cm_df = pd.DataFrame(
    cm,
    columns=["Predicted_0","Predicted_1"],
    index=["Actual_0","Actual_1"]
)

# ==========================
# Evaluation Sheet
# ==========================

eval_df = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "False Positive Rate",
        "True Positive Rate",
        "ROC-AUC"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        fpr_metric,
        tpr_metric,
        roc_auc
    ]
})

# ==========================
# ROC Sheet
# ==========================

roc_df = pd.DataFrame({
    "FPR":fpr,
    "TPR":tpr,
    "Threshold":thresholds
})

# ==========================
# Excel Export
# ==========================

def create_excel():

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine='openpyxl'
    ) as writer:

        raw_df.to_excel(
            writer,
            sheet_name="Raw_Data",
            index=False
        )

        train_df.to_excel(
            writer,
            sheet_name="Train_Data",
            index=False
        )

        test_df.to_excel(
            writer,
            sheet_name="Test_Data",
            index=False
        )

        cm_df.to_excel(
            writer,
            sheet_name="Confusion_Matrix"
        )

        eval_df.to_excel(
            writer,
            sheet_name="Model_Evaluation",
            index=False
        )

        roc_df.to_excel(
            writer,
            sheet_name="ROC_Curve_Data",
            index=False
        )

    output.seek(0)

    return output

excel_file = create_excel()

# ==========================
# Download Button
# ==========================

st.download_button(
    label="Download Excel Report",
    data=excel_file,
    file_name="Titanic_Model_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
