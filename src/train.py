import Config
import pandas as pd
from catboost import CatBoostRegressor
import pickle

df = pd.read_csv(Config.preprocess_data + 'Train_Preprocessed.csv')
col=pickle.load(open('bst_col_v2.pkl', 'rb'))
y = df['windmill_generated_power(kW/h)']
X = df.drop(['windmill_generated_power(kW/h)'], axis=1)
X=X[col]
# Training Model
model=CatBoostRegressor(verbose=False)
model.fit(X,y)

# Saving Model
pickle.dump(model, open(Config.model_ouput+'model.pkl', 'wb'))
