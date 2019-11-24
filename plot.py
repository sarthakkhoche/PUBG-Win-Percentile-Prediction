import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet, LassoLars
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train_V2.csv")
train.columns = ['Id','groupId','matchId','assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace','killPoints','kills','killStreaks','longestKill','matchDuration','matchType','maxPlace','numGroups','rankPoints','revives','rideDistance','roadKills','swimDistance','teamKills','vehicleDestroys','walkDistance','weaponsAcquired','winPoints','winPlacePerc']

train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
train['healsandboosts'] = train['heals'] + train['boosts']
train['totaldistance'] = train['rideDistance'] + train['swimDistance'] + train['walkDistance']
#train.describe()

train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
train['killPlace'] = train['killPlace']*((100 - train['playersJoined'])/100 + 1)

train['killswithoutmoving'] = ((train['totaldistance']==0) & (train['kills'] > 0))

print(train[train['killswithoutmoving'] == True].shape)
train.drop(train[train['killswithoutmoving'] == True].index, inplace=True)
print(train[train['kills'] > 40].shape)
print(train[(train['headshotKills']/train['kills'] == 1) & (train['kills'] > 12)].shape)

print(train[(train['totaldistance'] == train['swimDistance'])&(train['winPlacePerc'] > 0.80)].shape)
train.drop(train[(train['totaldistance'] == train['swimDistance'])&(train['winPlacePerc'] > 0.80)].index, inplace=True)
train.drop(train[train['kills'] > 40].index, inplace=True)

print(train[train['weaponsAcquired'] > 50].shape)
train.drop(train[train['weaponsAcquired'] > 50].index, inplace=True)
print(train[train['heals'] > 35].shape)
train.drop(train[train['heals'] > 35].index, inplace=True)

train[train['winPlacePerc'].isnull()]
train.drop(2744604, inplace=True)

print(train[(train['weaponsAcquired'] > 35) & (train['totaldistance'] == 0)].shape)
train.drop(train[(train['weaponsAcquired'] > 35) & (train['totaldistance'] == 0)].index, inplace=True)


print(train[(train['weaponsAcquired'] > 30) & (train['totaldistance'] < 50) & (train.kills > 30)].shape)
train.drop(train[(train['weaponsAcquired'] > 30) & (train['totaldistance'] < 50) & (train.kills > 30)].index, inplace=True)
train[train.longestKill > 950].shape
train.drop(train[train['longestKill'] > 950].index, inplace=True)

columns = ['killsNorm', 'damageDealtNorm', 'matchDurationNorm','healsandboosts', 'DBNOs', 'killPlace','walkDistance', 'winPoints'
          ,'weaponsAcquired', 'killStreaks', 'longestKill', 'teamKills', 'maxPlace', 'assists', 'revives'
          ,'numGroups']


X = train[columns]
y = train['winPlacePerc']

train_X, val_X, train_y, val_y = train_test_split(X,y, test_size=0.2, random_state=1)
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0, min_samples_leaf=3, max_features='sqrt')
regressor.fit(train_X, train_y)
predict = regressor.predict(val_X)
print(np.sqrt(metrics.mean_squared_error(val_y, predict)))
print(regressor.score(val_X, val_y))

features = train_X.columns
importances = regressor.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()















