import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. LOAD & BASIC CLEANING
# ---------------------------------------------------------
local_file = "default_credit_card_clients.xlsx"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

if os.path.exists(local_file):
    df = pd.read_excel(local_file)
else:
    df = pd.read_excel(url, header=1)
    df.to_excel(local_file, index=False)

rename_dict = {
    'LIMIT_BAL': 'credit_limit', 'SEX': 'gender', 'EDUCATION': 'education',
    'MARRIAGE': 'marital_status', 'AGE': 'age',
    'PAY_0': 'status_sep', 'PAY_2': 'status_aug', 'PAY_3': 'status_jul',
    'PAY_4': 'status_jun', 'PAY_5': 'status_may', 'PAY_6': 'status_apr',
    'BILL_AMT1': 'bill_sep', 'BILL_AMT2': 'bill_aug', 'BILL_AMT3': 'bill_jul',
    'BILL_AMT4': 'bill_jun', 'BILL_AMT5': 'bill_may', 'BILL_AMT6': 'bill_apr',
    'PAY_AMT1': 'paid_sep', 'PAY_AMT2': 'paid_aug', 'PAY_AMT3': 'paid_jul',
    'PAY_AMT4': 'paid_jun', 'PAY_AMT5': 'paid_may', 'PAY_AMT6': 'paid_apr',
    'default payment next month': 'default'
}
df.rename(columns=rename_dict, inplace=True)
if 'ID' in df.columns: df.drop('ID', axis=1, inplace=True)

# Create Labels for EDA only (matches your previous academic style)
df['edu_label'] = df['education'].map({1: 'Grad', 2: 'Uni', 3: 'HS', 4: 'Other', 5: 'Other', 6: 'Other', 0: 'Other'})
df['gender_label'] = df['gender'].map({1: 'Male', 2: 'Female'})

# ---------------------------------------------------------
# 2. DATA INTEGRITY & SUMMARY
# ---------------------------------------------------------
def academic_summary(df_in):
    numeric_df = df_in.select_dtypes(include=[np.number])
    summary = pd.DataFrame(index=numeric_df.columns)
    summary['Type'] = numeric_df.dtypes
    summary['Missing'] = numeric_df.isnull().sum()
    summary['Mean'] = numeric_df.mean().round(2)
    summary['Skewness'] = numeric_df.skew().round(2)
    summary['Kurtosis'] = numeric_df.kurt().round(2)
    return summary

print("--- Academic Data Integrity Report ---")
print(academic_summary(df))

# ---------------------------------------------------------
# 3. VISUAL EDA
# ---------------------------------------------------------

# A. Univariate: Target, Age, and Credit
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sns.countplot(x='default', data=df, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Target Balance (Default vs Non-Default)')

sns.histplot(df['age'], bins=30, kde=True, ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Age Distribution')

sns.histplot(df['credit_limit'], bins=30, kde=True, ax=axes[1, 0], color='salmon')
axes[1, 0].set_title('Credit Limit Distribution')

sns.countplot(x='edu_label', data=df, ax=axes[1, 1], palette='pastel')
axes[1, 1].set_title('Education Background')
plt.tight_layout()
plt.show()

# B. Bivariate: Features vs Default
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='default', y='age', data=df, palette='Set2')
plt.title('Age vs Default Status')

plt.subplot(1, 2, 2)
sns.boxplot(x='default', y='credit_limit', data=df, palette='Set2')
plt.title('Credit Limit vs Default Status')
plt.show()

# C. Multicollinearity Assessment (Crucial for Naive Bayes/LDA)
plt.figure(figsize=(12, 8))
bill_cols = [c for c in df.columns if 'bill' in c]
sns.heatmap(df[bill_cols].corr(), annot=True, cmap='viridis', fmt=".2f")
plt.title("Multicollinearity in Monthly Bill Amounts")
plt.show()

# D. Temporal Trends
bill_trend_cols = ['bill_apr', 'bill_may', 'bill_jun', 'bill_jul', 'bill_aug', 'bill_sep']
paid_trend_cols = ['paid_apr', 'paid_may', 'paid_jun', 'paid_jul', 'paid_aug', 'paid_sep']
months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

plt.figure(figsize=(10, 5))
plt.plot(months, df[bill_trend_cols].mean(), marker='o', label='Avg Bill')
plt.plot(months, df[paid_trend_cols].mean(), marker='s', label='Avg Paid')
plt.title('Financial Trends (April - September)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ---------------------------------------------------------
# 4. TRANSFORMATION PIPELINE
# ---------------------------------------------------------

# 4.1 Manual Grouping
monetary_cols = [
    'credit_limit', 'age', 
    'bill_sep', 'bill_aug', 'bill_jul', 'bill_jun', 'bill_may', 'bill_apr',
    'paid_sep', 'paid_aug', 'paid_jul', 'paid_jun', 'paid_may', 'paid_apr'
]
ordinal_cols = [
    'education', 'status_sep', 'status_aug', 'status_jul', 
    'status_jun', 'status_may', 'status_apr'
]
nominal_cols = ['gender', 'marital_status']

# 4.2 Cleaning & Encoding
df['education'] = df['education'].replace([0, 5, 6], 4)
df['marital_status'] = df['marital_status'].replace(0, 3)

# One-Hot Encode and Clean X
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
X = df_encoded.drop(columns=['default', 'edu_label', 'gender_label'], errors='ignore')
y = df_encoded['default']

# 4.3 Split (Avoid Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4.4 Selective Automatic Logging
SKEW_THRESHOLD = 0.75
logged_cols = []
train_skewness_before = X_train[monetary_cols].skew()

for col in monetary_cols:
    if abs(train_skewness_before[col]) > SKEW_THRESHOLD:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))
        logged_cols.append(col)

# 4.5 Standardization
scaler = StandardScaler()
# Final check to ensure we only scale the columns we actually want for the models
final_cols = monetary_cols + ordinal_cols + [c for c in X.columns if 'gender_' in c or 'marital_status_' in c]
X_train = X_train[final_cols]
X_test = X_test[final_cols]

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ---------------------------------------------------------
# 5. FINAL VERIFICATION
# ---------------------------------------------------------
print("\n--- Preprocessing Verification ---")
if logged_cols:
    comparison = pd.DataFrame({
        'Variable': logged_cols,
        'Skew_Before': train_skewness_before[logged_cols].values,
        'Skew_After': X_train[logged_cols].skew().values
    }).set_index('Variable')
    print(comparison.round(3))

print(f"\nFinal Feature Count: {X_train_scaled.shape[1]}")
print("Step 2 EDA Complete.")