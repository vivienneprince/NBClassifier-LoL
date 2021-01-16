# Naive Bayes Classifier Demo
A Naive Bayes Classifier implemented in python using only numpy and pandas packages. With small adjustments, this code can be used to preform NB classification on any dataset using only the numerical features (binary is ok). 


It is currently configured to predict League of Legends game outcomes using information from the first 10 minutes of the game.  

An .arff version of the LoL data file, as well as a weka model is included in the weka test folder if you want to validate the results using weka. 
 
# About the data
Each game has two teams (blue, red) and generally lasts between 20-50 mintues.  
  
Class: blueWins (0 if team blue loses, and 1 if they win)   
Omitted variables: gameID   
The rest of the features are data about the first 10 minutes of a game, and this is what is used to predict the winner.    
 
# Asumptions
We assume that all relevant variables are conditionally independent given the class, and that they are have equal effect on the outcome.  


# How it works
This algorithm calculates and records the mean and standard deviation for each relevant variable for each class in the training set.  
  
When given new data to classify, the algorithm will calculate the likilihoods of the variables taking those values given each class using the probablity density function:  
  
![p-density-function](images/p-density-function.png)  
  
The class with the largest resulting likelihood is what it will predict, in other words:  
  
![classifier](images/classifier.png)  
  
* note in my code I use notation Y = (y1,y2,...,yj,...,yk) and X = (x1,x2,...,xi,...,xn) instead of just y or C  
* I also took ln of each likelihood and summed instead since some values got really small
  

# Refrences:
**Data**:   
https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min  

**Resources**   
https://towardsdatascience.com/naive-bayes-classifier-81d512f50a7c  
https://en.wikipedia.org/wiki/Naive_Bayes_classifier  
https://www.geeksforgeeks.org/naive-bayes-classifiers/  
