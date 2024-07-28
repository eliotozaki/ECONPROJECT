import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
from pandas_profiling import ProfileReport

df = pd.read_csv("PandasTest.csv")
df.head(10)

df.columns
df.types

df['Number'] = pd.to_numeric(df['Number'])

df.groupby('State')
df.head(10)
df.describe()

df["State"][df['State']=="Utah"]="UT" 
df.head(10)

df.to_csv("PandaTestEdit.csv")