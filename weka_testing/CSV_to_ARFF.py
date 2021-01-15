import arff
import pandas as pd

df = pd.read_csv("toarfftest.csv")

arff.dump('LoLwekaTest.arff',
          df.values,
          relation='LoLDiamondData',
          names=df.columns)
