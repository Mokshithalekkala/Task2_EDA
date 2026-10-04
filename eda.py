import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Display first 5 rows
print("FIRST 5 ROWS:")
print(df.head())

# Dataset information
print("\nDATASET INFORMATION:")
df.info()

# Summary statistics
print("\nSUMMARY STATISTICS:")
print(df.describe())

# Missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Histogram - Age
plt.figure(figsize=(8, 5))
sns.histplot(df["age"].dropna(), kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Boxplot - Age
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"].dropna())
plt.title("Boxplot of Age")
plt.xlabel("Age")
plt.show()

# Boxplot - Fare
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"].dropna())
plt.title("Boxplot of Fare")
plt.xlabel("Fare")
plt.show()

# Pairplot
pair_data = df[["age", "fare", "pclass", "survived"]].dropna()

sns.pairplot(pair_data, hue="survived")
plt.show()

# Correlation matrix
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Plotly scatter plot
fig = px.scatter(
    df,
    x="age",
    y="fare",
    color="survived",
    title="Age vs Fare"
)

fig.show()

# EDA observations
print("\nEDA OBSERVATIONS:")
print("1. The histogram shows the distribution of passenger ages.")
print("2. Boxplots help identify possible outliers in age and fare.")
print("3. The pairplot shows relationships between numerical features.")
print("4. The correlation matrix shows relationships between numerical variables.")
print("5. The scatter plot shows the relationship between age and fare.")