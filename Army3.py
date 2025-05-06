import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


np.random.seed(42)
countries = [f"Country_{i}" for i in range(1, 21)]
weapon_types = ["Tanks", "Drones", "Aircraft", "Artillery", "Naval", "Small_Arms", "Missiles"]

# Define alliances
def assign_alliance(country):
    idx = int(country.split('_')[1])
    if 1 <= idx <= 5:
        return "NATO"
    elif 6 <= idx <= 10:
        return "BRICS"
    elif 11 <= idx <= 15:
        return "ASEAN"
    else:
        return "Non-Aligned"

data = []
for country in countries:
    alliance = assign_alliance(country)
    for weapon in weapon_types:
        row = {
            "Country": country,
            "Alliance": alliance,
            "Weapon_Type": weapon,
            "Count": np.random.randint(50, 10000),
            "Age": np.random.uniform(1, 40),
            "Modernization_Level": np.random.uniform(0, 1),
            "Operational_Readiness": np.random.uniform(0.5, 1),
            "Cost_per_Unit": np.random.uniform(100_000, 10_000_000),
            "Usage_Rate": np.random.uniform(0, 1)
        }
        data.append(row)

df = pd.DataFrame(data)

# ---- Step 2: Normalize features ----
features = ["Count", "Age", "Modernization_Level", "Operational_Readiness", "Cost_per_Unit", "Usage_Rate"]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# ---- Step 3: PCA ----
pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled)
df["PC1"] = components[:, 0]
df["PC2"] = components[:, 1]

# ---- Step 4: Plot with alliance color and country labels ----
plt.figure(figsize=(14, 10))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Alliance", style="Weapon_Type", palette="Set2", s=100, alpha=0.8)

# Add country labels at average location
centers = df.groupby("Country")[["PC1", "PC2"]].mean().reset_index()
for _, row in centers.iterrows():
    plt.text(row["PC1"] + 0.2, row["PC2"], row["Country"], fontsize=9, weight='bold', alpha=0.7)

plt.title("Global Army Weaponry Statistics by Alliance (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
