import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('Data/StudentsPerformance.csv')

print("=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)

# Display summary statistics for numeric columns
numeric_columns = ['math score', 'reading score', 'writing score']
print("\nSummary statistics for numeric columns:")
print(df[numeric_columns].describe())

print("\n" + "=" * 50)
print("CORRELATION MATRIX")
print("=" * 50)
print(df[numeric_columns].corr())

# Create a figure with subplots for histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribution of Scores', fontsize=16, fontweight='bold')

# Plot histograms for each score column
score_columns = ['math score', 'reading score', 'writing score']
colors = ['skyblue', 'lightcoral', 'lightgreen']

for i, (col, color) in enumerate(zip(score_columns, colors)):
    axes[i].hist(df[col], bins=20, color=color, alpha=0.7, edgecolor='black')
    axes[i].set_title(f'{col.title()} Distribution')
    axes[i].set_xlabel('Score')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('score_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Create bar chart comparing average math scores by gender
plt.figure(figsize=(10, 6))

# Calculate average math scores by gender
gender_math_avg = df.groupby('gender')['math score'].mean()

# Create bar plot
bars = plt.bar(gender_math_avg.index, gender_math_avg.values, 
               color=['lightblue', 'lightpink'], alpha=0.8, edgecolor='black')

# Add value labels on top of bars
for bar, value in zip(bars, gender_math_avg.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

plt.title('Average Math Score by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Average Math Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, max(gender_math_avg.values) + 5)

plt.tight_layout()
plt.savefig('math_score_by_gender.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional insights
print("\n" + "=" * 50)
print("ADDITIONAL INSIGHTS")
print("=" * 50)

print(f"\nAverage scores by gender:")
for gender in df['gender'].unique():
    gender_data = df[df['gender'] == gender]
    print(f"\n{gender.title()}:")
    print(f"  Math: {gender_data['math score'].mean():.2f}")
    print(f"  Reading: {gender_data['reading score'].mean():.2f}")
    print(f"  Writing: {gender_data['writing score'].mean():.2f}")

print(f"\nOverall statistics:")
print(f"  Total students: {len(df)}")
print(f"  Gender distribution:")
print(df['gender'].value_counts()) 