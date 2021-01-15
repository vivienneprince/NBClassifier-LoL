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
features = [feature for feature in df.columns.values.tolist() if feature not in ('blueWins', 'gameId')]

# ===================================================================
# Split data
# ===================================================================

# create train/test subsets
np.random.seed(0)
mask = np.random.rand(len(df)) < 0.9
train = df[mask]
test = df[~mask]

# ===================================================================
# Learn from train set
# ===================================================================

P_Y = []
distributions_XGivenY = []

for classtype in classes:
    class_data = train[train['blueWins'] == classtype]
    P_Y.append(len(class_data) / len(train))

    for x in features:
        feature_data = class_data[x]
        distributions_XGivenY.append([x, np.mean(feature_data), np.std(feature_data)])

# ===================================================================
# model validation using test set
# ===================================================================

validation_array = []
validation_score = 0


def likelihood(value, mu, sigma):
    a = 2 * np.pi * sigma ** 2
    b = (value - mu) / sigma
    return 1 / np.sqrt(a) * np.exp(-0.5 * b ** 2)


def validate_result(guess, index):
    if guess == test['blueWins'][index]: return 1
    else: return 0


for ind in test.index:

    # initiate
    P_yjGivenX = np.NINF
    prediction = 0

    for classtype in classes:

        P_xiGivenyj_array = []

        for feature, feature_mu, feature_sigma in distributions_XGivenY:
            feature_value = test[feature][ind]
            P_xiGivenyj_array.append(np.log(likelihood(feature_value, feature_mu, feature_sigma)))

        temp_P = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)

        if temp_P > P_yjGivenX:
            P_yjGivenX = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)
            prediction = classtype

    validation_array.append(validate_result(prediction, ind))

validation_score = np.mean(validation_array)
print(validation_score)
