import numpy as np
import pandas as pd

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
    P_Y.append(len(class_data) / len(train))  # get P(yj)

    class_feature_distributions = []

    for feature in features:
        feature_data = class_data[feature]
        class_feature_distributions.append([np.mean(feature_data), np.std(feature_data)])

    distributions_XGivenY.append(class_feature_distributions)

# convert to pd df
# this is in form: columns=classes, rows=features
distributions_XGivenY = pd.DataFrame(distributions_XGivenY, columns=[features]).transpose()

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
    if guess == test['blueWins'][index]:
        return 1
    else:
        return 0


for ind in test.index:

    # initiate
    P_yjGivenX = np.NINF
    prediction = 0

    for classtype in classes:

        class_distribution_data = distributions_XGivenY[classtype]

        P_xiGivenyj_array = []

        for feature in features:
            feature_data = class_distribution_data[feature]
            feature_value = test[feature][ind]
            feature_mu, feature_sigma = np.concatenate(feature_data.to_numpy()).ravel()

            P_xiGivenyj_array.append(np.log(likelihood(feature_value, feature_mu, feature_sigma)))

        temp_P = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)

        if temp_P > P_yjGivenX:
            P_yjGivenX = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)
            prediction = classtype

    validation_array.append(validate_result(prediction, ind))

validation_score = np.mean(validation_array)
print(validation_score)
