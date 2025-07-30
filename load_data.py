import pandas as pd

# Load the dataset
df = pd.read_csv('Data/StudentsPerformance.csv')

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display basic information about the dataset
print("\nDataset info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}") 