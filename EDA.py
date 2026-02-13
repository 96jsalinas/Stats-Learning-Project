import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. LOAD DATA (LOCAL CACHE OR DOWNLOAD)
# ---------------------------------------------------------
# Define the local filename and the source URL
local_file = "default_credit_card_clients.xlsx"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

if os.path.exists(local_file):
    print(f"Found local dataset '{local_file}'. Loading...")
    # If loading our saved local file, the headers are standard (row 0)
    df = pd.read_excel(local_file)
else:
    print("Local dataset not found. Downloading from UCI Archive... (this may take a moment)")
    # Note: The original UCI file has 2 header rows. Row 1 contains the actual variable names.
    df = pd.read_excel(url, header=1)
    
    print(f"Saving copy to '{local_file}' for future runs...")
    df.to_excel(local_file, index=False)

# ---------------------------------------------------------
# 2. DATA CLEANING & RENAMING
# ---------------------------------------------------------

# Drop the 'ID' column as it is just an index
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# Rename dictionary based on UCI documentation
# 1 = Sept, 2 = Aug, 3 = Jul, 4 = Jun, 5 = May, 6 = Apr
rename_dict = {
    # Demographics
    'LIMIT_BAL': 'credit_limit',
    'SEX': 'gender',
    'EDUCATION': 'education',
    'MARRIAGE': 'marital_status',
    'AGE': 'age',
    
    # Repayment Status (PAY_0 is Sept, PAY_2 is Aug, etc.)
    'PAY_0': 'status_sep',
    'PAY_2': 'status_aug',
    'PAY_3': 'status_jul',
    'PAY_4': 'status_jun',
    'PAY_5': 'status_may',
    'PAY_6': 'status_apr',
    
    # Bill Amounts
    'BILL_AMT1': 'bill_sep',
    'BILL_AMT2': 'bill_aug',
    'BILL_AMT3': 'bill_jul',
    'BILL_AMT4': 'bill_jun',
    'BILL_AMT5': 'bill_may',
    'BILL_AMT6': 'bill_apr',
    
    # Previous Payments
    'PAY_AMT1': 'paid_sep',
    'PAY_AMT2': 'paid_aug',
    'PAY_AMT3': 'paid_jul',
    'PAY_AMT4': 'paid_jun',
    'PAY_AMT5': 'paid_may',
    'PAY_AMT6': 'paid_apr',
    
    # Target
    'default payment next month': 'default'
}

df.rename(columns=rename_dict, inplace=True)

# Convert the target to a categorical type for better plotting
df['default'] = df['default'].astype('category')

print(f"Dataset Loaded & Processed Successfully!")
print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print("-" * 30)

# ---------------------------------------------------------
# 3. EDA PART 1: TARGET BALANCE CHECK
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='default', data=df, palette='viridis')
plt.title('Class Distribution: Non-Default (0) vs Default (1)')
plt.xlabel('Default Status')
plt.ylabel('Count')
plt.show()

# Calculate exact percentage
default_rate = df['default'].value_counts(normalize=True)[1] * 100
print(f"Default Rate: {default_rate:.2f}%")

# ---------------------------------------------------------
# 4. EDA PART 2: CORRELATION MATRIX
# ---------------------------------------------------------
# Why this matters:
# Naive Bayes assumes features are INDEPENDENT. 
# We can now see if 'bill_sep' is correlated with 'bill_aug'.
plt.figure(figsize=(12, 10))
# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_df.corr()

sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# ---------------------------------------------------------
# 5. EDA PART 3: NUMERICAL DISTRIBUTION (AGE)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
# Note: Column name changed from 'AGE' to 'age'
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.show()

# Display first few rows to verify the new names
print("First 5 rows of data with new names:")
print(df.head())