import pandas as pd
import numpy as np

df = pd.read_csv("high_diamond_ranked_10min.csv")

# ===================================================================
# Definitions
# ===================================================================

# define classes
# 0 = red wins
# 1 = blue wins

# even though this is just binary (win/ lose)
# lets just do both so it can be more easily scaled for more classes in the future
classes = np.sort(df['blueWins'].unique().tolist())

# define features
features = df.columns.values.tolist()

# ===================================================================
# Split data
# ===================================================================

# create train/test subsets
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]

# ===================================================================
# Learn from train set
# ===================================================================

P_Y = []
P_XGivenY = []

for classtype in classes:
    class_data = df[df['blueWins'] == classtype]
    P_Y.append(len(class_data) / len(df))

    for x in features:
        feature_data = class_data[x]
        print(feature_data)

print(P_Y)
print(P_XGivenY)

# now we test
