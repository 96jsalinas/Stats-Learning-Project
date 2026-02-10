import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD DATA DIRECTLY FROM SOURCE
# ---------------------------------------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

print("Downloading and loading dataset... (this may take a moment)")
# Note: The file has 2 header rows. Row 1 contains the actual variable names.
df = pd.read_excel(url, header=1)

# ---------------------------------------------------------
# 2. BASIC CLEANING
# ---------------------------------------------------------
# Rename the target column to something easier to type
df.rename(columns={'default payment next month': 'Default'}, inplace=True)

# Drop the 'ID' column as it is just an index and has no predictive power
df.drop('ID', axis=1, inplace=True)

# Convert the target to a categorical type for better plotting
df['Default'] = df['Default'].astype('category')

print(f"Dataset Loaded Successfully!")
print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
print("-" * 30)

# ---------------------------------------------------------
# 3. EDA PART 1: TARGET BALANCE CHECK
# ---------------------------------------------------------
# Why this matters for your project:
# If 'Default' (1) is very rare, your Probabilistic model might just guess 'No Default' (0) 
# every time and still get high accuracy. You need to know this baseline.
plt.figure(figsize=(6, 4))
sns.countplot(x='Default', data=df, palette='viridis')
plt.title('Class Distribution: Non-Default (0) vs Default (1)')
plt.xlabel('Default Status')
plt.ylabel('Count')
plt.show()

# Calculate exact percentage
default_rate = df['Default'].value_counts(normalize=True)[1] * 100
print(f"Default Rate: {default_rate:.2f}%")

# ---------------------------------------------------------
# 4. EDA PART 2: CORRELATION MATRIX
# ---------------------------------------------------------
# Why this matters:
# Naive Bayes assumes features are INDEPENDENT. 
# If 'BILL_AMT1' is highly correlated with 'BILL_AMT2' (which it likely is),
# you have a great talking point about why Naive Bayes is theoretically 'naive' here.
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
# Why this matters:
# Check if key variables are normally distributed or skewed.
plt.figure(figsize=(8, 5))
sns.histplot(df['AGE'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.show()

# Display first few rows to verify structure
print("First 5 rows of data:")
print(df.head())