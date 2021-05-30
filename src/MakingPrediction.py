import Config
import pickle
import pandas as pd

# Loading Test data
df_test = pd.read_csv(Config.test_data)

# Adding date-time features
df_test['datetime'] = pd.to_datetime(df_test['datetime'].values)
df_test['month'] = pd.DatetimeIndex(df_test['datetime']).month
df_test['year'] = pd.DatetimeIndex(df_test['datetime']).year
df_test['week'] = pd.DatetimeIndex(df_test['datetime']).week
df_test['hour'] = pd.DatetimeIndex(df_test['datetime']).hour
df_test['min'] = pd.DatetimeIndex(df_test['datetime']).minute
df_test['sec'] = pd.DatetimeIndex(df_test['datetime']).second

# Loading Encoder
Encoders = pickle.load(open(Config.encoder + 'Encoders.pkl', 'rb'))


# preprocessing
def Preprocessing(df):
    def median_onlyPositive(col1):
        a = df[col1][df[col1] > 0].median()
        return a

    correcttion_col = ['wind_speed(m/s)', 'atmospheric_temperature(°C)', 'shaft_temperature(°C)',
                       'gearbox_temperature(°C)',
                       'engine_temperature(°C)', 'atmospheric_pressure(Pascal)', 'area_temperature(°C)',
                       'windmill_body_temperature(°C)',
                       'resistance(ohm)', 'rotor_torque(N-m)', 'blade_length(m)', 'windmill_height(m)']
    for col in correcttion_col:
        df[col] = df[col].apply(lambda x: median_onlyPositive(col) if x < 0 else x)

    df = df.drop(['tracking_id', 'datetime'], axis=1)
    num_col = [col for col in df.columns if df[col].dtype != 'O']
    list(set(df.columns) - set(num_col))
    # Handling Missing Numerical value
    for col in num_col:
        df[col] = df[col].fillna(df[col].mean())

    # Handling Missing Categorical columns
    df['turbine_status'] = df['turbine_status'].fillna('Missing_info')
    df['cloud_level'] = df['cloud_level'].fillna('Missing_info')

    # For the time being- I am removing date columns from the dataset. I will consider while improving the model
    # --------------
    return df


df_test = Preprocessing(df_test)

num_col = [col for col in df_test.columns if df_test[col].dtype != 'O']
cat_col = list(set(df_test.columns) - set(num_col))

col_OHE = ['']
col_LE = list(set(cat_col) - set(col_OHE))

for col in col_LE:
    df_test[col] = Encoders['L_enc_' + str(col)].transform(df_test[col])
    print(col)
# Drop the Un-necessary feature
df_test = df_test.drop(['sec', 'min'], axis=1)

# Loading Model
model = pickle.load(open(Config.model_ouput + 'model.pkl', 'rb'))
col=pickle.load(open('bst_col_v2.pkl', 'rb'))
pred = model.predict(df_test)

df_test = pd.read_csv(Config.test_data)[col]
final_dict = {'tracking_id': df_test.tracking_id.values, 'datetime': df_test.datetime.values,
              'windmill_generated_power(kW/h)': pred}
final_sub = pd.DataFrame(final_dict)
final_sub.to_csv(Config.submission + 'Submission.csv', index=False)
