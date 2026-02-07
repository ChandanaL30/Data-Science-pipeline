#  Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#  Step 2: Load Data
print(" Loading the Iris dataset...")
data = sns.load_dataset("iris")  # small built-in dataset
print(" Data loaded successfully!\n")

print("First 5 rows of data:")
print(data.head(), "\n")

# Step 3: Clean Data
print(" Checking for missing values...")
print(data.isnull().sum(), "\n")

# Handle missing values (if any)
data = data.dropna()

# Rename columns for better readability
data.rename(columns={"sepal_length": "Sepal Length",
                     "sepal_width": "Sepal Width",
                     "petal_length": "Petal Length",
                     "petal_width": "Petal Width"}, inplace=True)

print(" Data cleaned successfully!\n")

# Step 4: Analyze Data
print(" Basic Statistics:")
print(data.describe(), "\n")

print(" Count of each species:")
print(data['species'].value_counts(), "\n")

#  Step 5: Visualize Data
# Histogram of Sepal Length
plt.figure(figsize=(6, 4))
plt.hist(data["Sepal Length"], bins=15)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Pairplot for feature comparison
sns.pairplot(data, hue="species")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(data.drop(columns=["species"]).corr(),annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

#  Step 6: Present Results
print(" Insights:")
print("- Iris dataset has 150 samples and 3 species: setosa, versicolor, virginica.")
print("- Sepal and petal measurements can distinguish between species.")
print("- Petal length and petal width are highly correlated (r > 0.95).")
print("- Setosa flowers generally have smaller petals and sepals compared to others.")
print("\n Data Science Pipeline Completed Successfully!")