import pandas as pd
import pickle
import Config
import preprocessing_utils as pu
# Reading data
df = pd.read_csv(Config.train_data, parse_dates=True)

# Adding Data-Time Columns
df['datetime'] = pd.to_datetime(df['datetime'].values)

df['month'] = pd.DatetimeIndex(df['datetime']).month
df['year'] = pd.DatetimeIndex(df['datetime']).year
df['week'] = pd.DatetimeIndex(df['datetime']).week
df['hour'] = pd.DatetimeIndex(df['datetime']).hour
df['min'] = pd.DatetimeIndex(df['datetime']).minute
df['sec'] = pd.DatetimeIndex(df['datetime']).second

# Dropping Unecessary Features
df = df.drop(['tracking_id', 'datetime'], axis=1)

# Cat_Num Columns
num_col = [col for col in df.columns if df[col].dtype != 'O']
cat_col = list(set(df.columns) - set(num_col))


# Correction_In_Data
def median_onlyPositive(col):
    a = df[col][df[col] > 0].median()
    return a


correcttion_col = ['wind_speed(m/s)', 'atmospheric_temperature(°C)', 'shaft_temperature(°C)', 'gearbox_temperature(°C)',
                   'engine_temperature(°C)', 'atmospheric_pressure(Pascal)', 'area_temperature(°C)',
                   'windmill_body_temperature(°C)', 'resistance(ohm)', 'rotor_torque(N-m)', 'blade_length(m)',
                   'windmill_height(m)']
for col in correcttion_col:
    df[col] = df[col].apply(lambda x: median_onlyPositive(col) if x < 0 else x)

# Dealing_Missing_Numerical_Values
for col in num_col:
    df[col] = df[col].fillna(df[col].mean())

# Dealing_With_Categorical_Values
df['turbine_status'] = df['turbine_status'].fillna('Missing_info')
df['cloud_level'] = df['cloud_level'].fillna('Missing_info')

df_final, Encoders=pu.Cat2Num(df)

# Saving Preprocessed_Data
df_final.to_csv(Config.preprocess_data+'Train_Preprocessed.csv', index=False)
pickle.dump(Encoders, open(Config.encoder+'Encoders.pkl', 'wb'))
