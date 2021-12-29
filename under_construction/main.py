import numpy as np
import pandas as pd


PATH_TO_DATA_FILE = "high_diamond_ranked_10min.csv"
CLASS_COLNAME = "blueWins"
OMIT_FEATURES = ["gameId"]
TRAIN_SIZE = 0.9  # Ratio train:data
np.random.seed(0)  # Set random seed


def likelihood(value, mu, sigma):
    """Probability density function. Function image: https://wikimedia.org/api/rest_v1/media/math/render/svg/c9167a4f19898b676d4d1831530a8ff1246d33ab

    Args:
        value (float): Value to calcualte likelihood from normal distribution
        mu (float): Mean of distribution 
        sigma (float): Standard deviation of distribution 

    Returns:
        float: Likelihood of value given N(mu, sigma)
    """
    a = 2 * np.pi * sigma ** 2
    b = (value - mu) / sigma
    return 1 / np.sqrt(a) * np.exp(-0.5 * b ** 2)


def validate_result(guess, index, test):
    # compares guess with actual value from data
    if guess == test['blueWins'][index]: return 1
    else: return 0
    

class NB_model:
    # Notation:
    # Y = (y1,y2,..,yk) classes
    # X = (x1,x2,...,xn) features

    def __init__(self):
        self.distributions_XGivenY = pd.DataFrame()  # P(X|Y)
        self.P_Y = []  # P(Y)

    def train(self, traindata, classname, classes, features):
        for classtype in classes:
            class_data = traindata[traindata[classname] == classtype]  # grab data for yj
            self.P_Y.append(len(class_data) / len(traindata))  # get P(yj)
            class_feature_distributions = []  # temp variable to store feature distributions for yj

            for feature in features:
                feature_data = class_data[feature]  # grab data for xi|yj

                class_feature_distributions.append([np.mean(feature_data), np.std(feature_data)])

            # each class gets their own array for feature distributions
            self.distributions_XGivenY.append(class_feature_distributions)

        # convert feature|class distribution data to pd df
        # this is in form: columns=classes, rows=features
        self.distributions_XGivenY = pd.DataFrame(self.distributions_XGivenY, columns=[features]).transpose()

    def predict():
        #TODO: PREDICTION
        prediction = 0
        return prediction

    def test_accuracy(testdata):
        validation_array = []
        np.mean(validation_array)









def test_nb_accuracy(filepath, class_colname, omitted_features):
    df = pd.read_csv("high_diamond_ranked_10min.csv")



if __name__ == "__main__":
    test_nb_accuracy(PATH_TO_DATA_FILE, CLASS_COLNAME, OMIT_FEATURES)