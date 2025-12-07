import pandas as pd

df_sample = pd.read_csv('dataset/test.csv', nrows=100)
df_sample.to_csv('dataset/sample1.csv', index=False)