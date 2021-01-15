import arff
import pandas as pd

df = pd.read_csv("high_diamond_ranked_10min.csv")

arff.dump('high_diamond_ranked_10min.arff',
          df.values,
          relation='LoLDiamondData',
          names=df.columns)
