import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('UM_C19_2021.csv')

# Print basic information
print("=== DATASET BASIC INFORMATION ===")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)

# Print column analysis
print("\n=== COLUMN ANALYSIS ===")

# Date column analysis
df['Date'] = pd.to_datetime(df['Date'])
print(f"\nDate range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"Time period: {(df['Date'].max() - df['Date'].min()).days + 1} days")

# Type column analysis
print("\nUnique values in 'Type' column:")
print(df['Type'].value_counts())

# Residence column analysis
print("\nUnique values in 'residence' column:")
print(df['residence'].value_counts())

# Numeric columns analysis
print("\n=== NUMERICAL DATA SUMMARY ===")
print("Summary statistics for test counts:")
print(df[['Positive', 'Negative']].describe())

# Calculate additional metrics
total_tests = df['Positive'].sum() + df['Negative'].sum()
total_positive = df['Positive'].sum()
total_negative = df['Negative'].sum()
positivity_rate = total_positive / total_tests * 100 if total_tests > 0 else 0

print(f"\nTotal tests: {total_tests}")
print(f"Total positive tests: {total_positive}")
print(f"Total negative tests: {total_negative}")
print(f"Overall positivity rate: {positivity_rate:.2f}%")

# Check for missing values
print("\n=== MISSING VALUES CHECK ===")
print(df.isnull().sum())

# Preview some sample data
print("\n=== SAMPLE DATA (first 5 rows) ===")
print(df.head())