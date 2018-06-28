import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('mopacspring2018.csv')
df['sum'] = np.sum(df.loc[:], axis=1)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['time'] = pd.to_datetime(df['time'])





fig, ax = plt.subplots()
plt.plot_date(df['time'], df['sum'])
ax.set_title('Mopac Toll Prices')
# plt.show()

X, y = np.array(df['datetime'].tolist()), np.array(df['sum'].tolist()) # make these list like arrays
d = np.array(df['datetime'].apply(lambda x: x.toordinal()).tolist())
hour = np.array(df['datetime'].apply(lambda x: x.hour).tolist())
month = np.array(df['datetime'].apply(lambda x: x.month).tolist())

# print(X.shape, y.shape)
# classifier = KNeighborsClassifier()
# print(df.head())

X1 = []
for i in range(len(X)):
	X1.append(np.array([d[i],hour[i],month[i]])) # eventually we can add more input data here
X1 = np.array(X1)
print(X1.shape)

train_X, test_X, train_y, test_y = train_test_split(X1, y, test_size=0.5, random_state=123)
print("Labels for training and testing data")
print(train_y.shape)
print(test_y.shape)

# df.to_csv('mopac_sum.csv')


k = 20 # ?? I dont know what is best to use yet
knn = KNeighborsRegressor(n_neighbors=k) # use regressor for this kinda data
knn.fit(train_X,train_y)

######
# from here on you can look at the metrics and see how it looks, like what is the corrolation?

# you could also try to predict the future by feeding an array of the shape (day, hour, month)

# somthing like
future_time = [[5,9,7]] # 5th of july at 9 am
future_time = np.array(future_time)

y_pred = knn.predict(future_time)
print(y_pred)

plt.show()
