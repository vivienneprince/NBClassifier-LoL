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
np.random.seed(2)
mask = np.random.rand(len(df)) < 0.8
train = df[mask]
test = df[~mask]

# ===================================================================
# Learn from train set
# ===================================================================

P_Y = []
distributions_XGivenY = []

for classtype in classes:
    class_data = df[df['blueWins'] == classtype]
    P_Y.append(len(class_data) / len(df))

    for x in features:
        if x in ['blueWins', 'gameId']:
            continue
        feature_data = class_data[x]
        distributions_XGivenY.append([np.mean(feature_data), np.std(feature_data)])

print(P_Y)
print(distributions_XGivenY)

# ===================================================================
# model validation using test set
# ===================================================================

