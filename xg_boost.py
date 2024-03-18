import pandas as pd
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder

# Read the data
df = pd.read_csv('https://raw.githubusercontent.com/sg-tarek/HTML-CSS-and-JavaScript/main/Model%20Deployment/data.csv', index_col='Unnamed: 0')

# Fit the data to an ML model
x = df[['temperature', 'humidity', 'windspeed']]
y = list(set(df['count']))

model = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
le = LabelEncoder()

y_train_encoded = le.fit_transform(y)
model.fit(x, y)

# Save the model
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))