import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


train = pd.read_csv("train_sampled.csv")
train.columns = ['Id','groupId','matchId','assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace','killPoints','kills','killStreaks','longestKill','matchDuration','matchType','maxPlace','numGroups','rankPoints','revives','rideDistance','roadKills','swimDistance','teamKills','vehicleDestroys','walkDistance','weaponsAcquired','winPoints','winPlacePerc']


#creating a new column
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
train['healsandboosts'] = train['heals'] + train['boosts']
train['totaldistance'] = train['rideDistance'] + train['swimDistance'] + train['walkDistance']

#finding outliers
train['killswithoutmoving'] = ((train['totaldistance']==0) & (train['kills'] > 0))

print(train.drop(train[train['killswithoutmoving'] == True].index, inplace=True))





#normalizing rows based on the new column playersJoined
#train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
#train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
#train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
#train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)

#columns = ['playersJoined', 'killsNorm', 'damageDealtNorm', 'maxPlaceNorm', 'matchDurationNorm', 'assists', 'healsandboosts', 'DBNOs', 'killPlace', 'walkDistance', 'winPoints']
columns = ['playersJoined', 'kills', 'damageDealt', 'maxPlace', 'matchDuration', 'assists', 'healsandboosts', 'DBNOs', 'killPlace', 'walkDistance', 'winPoints']
X = train[columns]
y = train['winPlacePerc']

train_X, val_X, train_y, val_y = train_test_split(X,y, test_size=0.2, random_state=1)
model = LinearRegression()
model.fit(train_X, train_y)
pred = model.predict(val_X)
print(np.sqrt(metrics.mean_squared_error(val_y, pred)))
print(model.score(val_X, val_y))
