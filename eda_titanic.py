
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs("eda_output", exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("titanic.csv")

# -----------------------------
# Data overview
# -----------------------------
print("\n--- Data Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Value Counts ---")
print("Pclass:\n", df['Pclass'].value_counts())
print("Sex:\n", df['Sex'].value_counts())
print("Survived:\n", df['Survived'].value_counts())

# -----------------------------
# Age Distribution
# -----------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["Age"].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_output/hist_Age.png")
plt.close()

# -----------------------------
# Fare Distribution
# -----------------------------
plt.figure(figsize=(6, 4))
plt.boxplot(df["Fare"].dropna(), vert=False)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.tight_layout()
plt.savefig("eda_output/box_Fare.png")
plt.close()

# -----------------------------
# Pairplot (numeric features)
# -----------------------------
sns.pairplot(df[['Age','Fare','Pclass','Survived']], hue='Survived', corner=True)
plt.savefig("eda_output/pairplot.png")
plt.close()

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(6, 4))
corr = df[["Survived", "Pclass", "Age", "Fare"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_output/correlation_heatmap.png")
plt.close()

# -----------------------------
# Write Observations
# -----------------------------
observations = """
1. Most passengers were aged 20–40.
2. Fare has significant outliers; higher fare passengers had better survival chances.
3. Pclass negatively correlates with survival.
4. Females had higher survival rate than males.
5. Age and Fare show some correlation with survival.
6. Higher class passengers (Pclass=1) generally paid higher fares and had better survival.
"""

with open("eda_output/observations.txt", "w") as f:
    f.write(observations)

# -----------------------------
# Print Completion Message
# -----------------------------
print("EDA completed successfully. Check the 'eda_output' folder for plots and observations.")
