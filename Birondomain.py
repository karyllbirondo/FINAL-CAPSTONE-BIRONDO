# Import libraries
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("WSL_2019_Mock_Data.csv")

# Check and clean data
df = df.dropna()  # Not needed here, data is already clean
print(f"Number of data points: {len(df)}")  # Must be at least 100

# NumPy Operations
Event_Score = df["Event_Score"].to_numpy()
Waves_Surfed = df["Waves_Surfed"].to_numpy()    
Best_Wave_Score = df["Best_Wave_Score"].to_numpy()
Prize_Money_Won = df["Prize_Money_Won"].to_numpy()
Surfer_ID = df["Surfer_ID"].to_numpy()


mean_score = np.mean(Event_Score)
std_waves = np.std(Waves_Surfed)
median_best_wave = np.median(Best_Wave_Score)
max_prize = np.max(Prize_Money_Won)
mask = df["Event_Score"] > mean_score
above_avg_surfer_ids = df.loc[mask, "Surfer_ID"].values

print("NumPy Analysis:")
print(f"Mean Event Score: {mean_score}")
print(f"Standard Deviation of Waves Surfed: {std_waves}")
print(f"Median Best Wave Score: {median_best_wave}")
print(f"Maximum Prize Money Won: {max_prize}")
print(f"Number of Surfers Above Average Score: {len(above_avg_surfer_ids)}")

# SciPy Operation - Pearson Correlation
corr, pval = stats.pearsonr(df["Event_Score"], df["Waves_Surfed"])
print("\nSciPy Pearson Correlation:")
print(f"Correlation Coefficient: {corr}, p-value: {pval}")

# Statsmodels Operation - Linear Regression
X = df[["Waves_Surfed", "Best_Wave_Score"]]
X = sm.add_constant(X)  # add constant/intercept
y = df["Event_Score"]
model = sm.OLS(y, X).fit()
print("\nStatsmodels Linear Regression Summary:")
print(model.summary())

# Visualization 1 - Histogram of Event Scores
plt.figure(figsize=(8, 5))
sns.histplot(df["Event_Score"], kde=True, color="skyblue")
plt.title("Distribution of Event Scores")
plt.xlabel("Event Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Visualization 2 - Boxplot of Prize Money by Country (Top 5)
top_countries = df["Country"].value_counts().nlargest(5).index
filtered_df = df[df["Country"].isin(top_countries)]

plt.figure(figsize=(8, 5))
sns.boxplot(data=filtered_df, x="Country", y="Prize_Money_Won", hue="Country", palette="Set2", legend=False)
plt.title("Prize Money Distribution by Country (Top 5)")
plt.xlabel("Country")
plt.ylabel("Prize Money Won")
plt.tight_layout()
plt.show()

# Regression plot: Waves Surfed vs Event Score
plt.figure(figsize=(8, 6))
sns.regplot(
    data=df,
    x="Waves_Surfed",
    y="Event_Score",
    scatter_kws={"color": "blue", "s": 40},  # Blue points, slightly larger
    line_kws={"color": "red", "linewidth": 2},  # Bold red regression line
    ci=95  # Confidence interval (default)
)

sns.set_style("whitegrid")  # or "ticks"


plt.title("Correlation between Waves Surfed and Event Score", fontsize=14)
plt.xlabel("Waves Surfed", fontsize=12)
plt.ylabel("Event Score", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
