import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm  


PATH_TO_DATA_FILE = "high_diamond_ranked_10min.csv"
CLASS_COLNAME = "blueWins"
OMIT_FEATURES = "gameId"
TRAIN_SIZE = 0.9  # Ratio train:data
np.random.seed(0)  # Set random seed


df = pd.read_csv(PATH_TO_DATA_FILE)

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
classes = np.sort(df[CLASS_COLNAME].unique().tolist())

# define features
# in our case we want to omit the class column and ID data from the training
features = [feature for feature in df.columns.values.tolist() if feature not in (CLASS_COLNAME, OMIT_FEATURES)]

# ===================================================================
# Split data
# ===================================================================

# create train/test subsets
mask = np.random.rand(len(df)) < TRAIN_SIZE
train = df[mask]
test = df[~mask]

# ===================================================================
# Learn from train set
# ===================================================================

# Y = (y1,y2,..,yk) classes
# X = (x1,x2,...,xn) features

P_Y = []
distributions_XGivenY = []

def get_dist(feature_data):
    return [np.mean(feature_data), np.std(feature_data)]

for classtype in classes:
    class_data = train[train[CLASS_COLNAME] == classtype]  # grab data for yj
    P_Y.append(len(class_data) / len(train))  # get P(yj)

    # class_feature_distributions = []  # temp variable to store feature distributions for yj
    # for feature in features:
    #     feature_data = class_data[feature]  # grab data for xi|yj
    #     class_feature_distributions.append([np.mean(feature_data), np.std(feature_data)])
    class_feature_distributions = Parallel(n_jobs=-1)(
        delayed(get_dist)(class_data[feature]) for feature in tqdm(features , desc="Training model for class {}".format(classtype)))

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
    if guess == test[CLASS_COLNAME][index]: return 1
    else: return 0


def predict(ind):
    P_yjGivenX = np.NINF  # P(yj|X)
    prediction = 0  # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]

    for classtype in classes:

        class_distribution_data = distributions_XGivenY[classtype]  # grab feature distribution data for yj

        P_xiGivenyj_array = []  # temp array to store log(P(xi|yj)) values

        for feature in features:
            feature_data = class_distribution_data[feature]  # grab xi distribution data
            feature_mu, feature_sigma = feature_data
            feature_value = test[feature][ind]  # grab instance xi data
            # calculate log(P(xi|yj))
            P_xiGivenyj_array.append(np.log(likelihood(feature_value, feature_mu, feature_sigma)))

        # calculate log( P(yj)*PROD(i=1,n)(P(xi|yj) )
        temp_P = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)

        if temp_P > P_yjGivenX:
            # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]
            P_yjGivenX = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)
            prediction = classtype

    return validate_result(prediction, ind)


# for ind in test.index:

#     # initiate
#     P_yjGivenX = np.NINF  # P(yj|X)
#     prediction = 0  # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]

#     for classtype in classes:

#         class_distribution_data = distributions_XGivenY[classtype]  # grab feature distribution data for yj

#         P_xiGivenyj_array = []  # temp array to store log(P(xi|yj)) values

#         for feature in features:
#             feature_data = class_distribution_data[feature]  # grab xi distribution data
#             feature_mu, feature_sigma = feature_data
#             feature_value = test[feature][ind]  # grab instance xi data
#             # calculate log(P(xi|yj))
#             P_xiGivenyj_array.append(np.log(likelihood(feature_value, feature_mu, feature_sigma)))

#         # calculate log( P(yj)*PROD(i=1,n)(P(xi|yj) )
#         temp_P = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)

#         if temp_P > P_yjGivenX:
#             # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]
#             P_yjGivenX = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)
#             prediction = classtype

#     validation_array.append(validate_result(prediction, ind))  # model validation
validation_array = Parallel(n_jobs=-1)(
        delayed(predict)(ind) for ind in tqdm(test.index , desc="Testing model performance".format(classtype)))

validation_score = np.mean(validation_array)  # prediction success rate. with np.rand seed = 0, success rate is ~75%
print("NB model accuracy: {:.5%}".format(validation_score))
