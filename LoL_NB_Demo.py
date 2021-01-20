import numpy as np
import pandas as pd

df = pd.read_csv("high_diamond_ranked_10min.csv")

# ===================================================================
# Definitions
# ===================================================================

# define classes
# LoL dataset classes definition:
# 0 = red wins
# 1 = blue wins
# --------------
# even though this is just binary (win/ lose)
# lets just do both so it can be easily scaled for more classes in the future
classes = np.sort(df['blueWins'].unique().tolist())

# define features
# in our case we want to omit the class column and ID data from the training
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

# Y = (y1,y2,..,yk) classes
# X = (x1,x2,...,xn) features

P_Y = []
distributions_XGivenY = []

for classtype in classes:
    class_data = train[train['blueWins'] == classtype]  # grab data for yj
    P_Y.append(len(class_data) / len(train))  # get P(yj)

    class_feature_distributions = []  # temp variable to store feature distributions for yj

    for feature in features:
        feature_data = class_data[feature]  # grab data for xi|yj

        class_feature_distributions.append([np.mean(feature_data), np.std(feature_data)])

    # each class gets their own array for feature distributions
    distributions_XGivenY.append(class_feature_distributions)

# convert feature|class distribution data to pd df
# this is in form: columns=classes, rows=features
distributions_XGivenY = pd.DataFrame(distributions_XGivenY, columns=[features]).transpose()

# ===================================================================
# model validation using test set
# ===================================================================

validation_array = []  # store if model was correct or incorrect
validation_score = 0  # mean of validation array


def likelihood(value, mu, sigma):
    # probability density function
    # image: https://wikimedia.org/api/rest_v1/media/math/render/svg/c9167a4f19898b676d4d1831530a8ff1246d33ab
    a = 2 * np.pi * sigma ** 2
    b = (value - mu) / sigma
    return 1 / np.sqrt(a) * np.exp(-0.5 * b ** 2)


def validate_result(guess, index):
    # compares guess with actual value from data
    if guess == test['blueWins'][index]: return 1
    else: return 0


for ind in test.index:

    # initiate
    P_yjGivenX = np.NINF  # P(yj|X)
    prediction = 0  # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]

    for classtype in classes:

        class_distribution_data = distributions_XGivenY[classtype]  # grab feature distribution data for yj

        P_xiGivenyj_array = []  # temp array to store log(P(xi|yj)) values

        for feature in features:
            feature_data = class_distribution_data[feature]  # grab xi distribution data
            feature_mu, feature_sigma = np.concatenate(feature_data.to_numpy()).ravel()

            feature_value = test[feature][ind]  # grab instance xi data

            # calculate log(P(xi|yj))
            P_xiGivenyj_array.append(np.log(likelihood(feature_value, feature_mu, feature_sigma)))

        # calculate log( P(yj)*PROD(i=1,n)(P(xi|yj) )
        temp_P = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)

        if temp_P > P_yjGivenX:
            # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]
            P_yjGivenX = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)
            prediction = classtype

    validation_array.append(validate_result(prediction, ind))  # model validation

validation_score = np.mean(validation_array)  # perdiction success rate. with np.rand seed = 0, success rate is ~75%
print(validation_score)
