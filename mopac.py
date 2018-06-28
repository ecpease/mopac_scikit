import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('mopacspring2018.csv')
df['sum'] = np.sum(df.loc[:], axis=1)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['time'] = pd.to_datetime(df['time'])
# print(df['time'])

fig, ax = plt.subplots()
plt.plot_date(df['time'], df['sum'])
ax.set_title('Mopac Toll Prices')
# plt.show()

X, y = df['datetime'], df['sum']
print(X.shape, y.shape)
classifier = KNeighborsClassifier()
# print(df.head())
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=123)
print("Labels for training and testing data")
print(train_y)
print(test_y)

# df.to_csv('mopac_sum.csv')

# plt.show()