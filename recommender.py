import tensorflow
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pandas as pd
import math
import statistics
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error

df = pd.read_csv("IMDB-Movie-Data.csv")
df.head()

df.tail()

rate = df.reset_index()['Rating']
print(rate)

c = statistics.mean(rate)
print(c)

m =  np.quantile(rate, .50)
print(m)

q_movies = df.copy().loc[df['Rating'] >= m]
q_movies.shape()

df.shape()

def weighted_rating(x, m=m, c=c):
    v = x['Metascore']
    R = x['Metascore']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * c)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values('score',  ascending=False)
print( q_movies[['Title','Rating', 'score']].head(35) )



