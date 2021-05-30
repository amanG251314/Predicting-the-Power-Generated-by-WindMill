from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def Cat2Num(dff):
    df1 = dff.copy()

    cat_col = [col for col in df1.columns if df1[col].dtype == 'O']
    num_col = list(set(df1.columns) - set(cat_col))
    cat_ord = ['turbine_status','cloud_level']  # Ordinal category
    cat_nom = list(set(cat_col) - set(cat_ord))  # Nominal category
    #cat_nom = list(set(cat_nom))

    # Further classify nominal as having low or high cardinality
    low_cardinality_nom = [col for col in cat_nom if df1[col].nunique() < 10]
    high_cardinality_nom = list(set(cat_nom) - set(low_cardinality_nom))

    # Ordinal & High Cardinality 'Label Encoder' otherwise OneHotEncoder
    col_LE = cat_ord + high_cardinality_nom
    ##print(col_LE)
    col_OHE = list(set(cat_col) - set(col_LE))
    print(col_OHE)
    df_LE = df1[col_LE].copy()
    df_num = df1[num_col].copy()
    Encoders = {}
    for col in col_LE:
        Encoders['L_enc_' + str(col)] = LabelEncoder()
        df_LE[col] = Encoders['L_enc_' + str(col)].fit_transform(df1[col])

    Encoders['OH_enc_'] = OneHotEncoder(handle_unknown='ignore', sparse=False)
    dataset = pd.DataFrame(Encoders['OH_enc_'].fit_transform(df1[col_OHE]))
    dataset.index = df1.index
    dataset.columns = Encoders['OH_enc_'].get_feature_names(df1[col_OHE].columns.tolist())

    df_final = (pd.concat([df_num, dataset, df_LE], axis=1))

    return df_final, Encoders


def SpaceRemove(d):
    d = d.rename(columns={'age': 'age',\
                            ' workclass': 'workclass',\
                            ' fnlwgt': 'final_weight',\
                            ' education': 'education',\
                            ' education-num': 'education_num',\
                            ' marital-status': 'marital_status',\
                            ' occupation': 'occupation',\
                            ' relationship': 'relationship',\
                            ' race': 'race',\
                            ' sex': 'sex',\
                            ' capital-gain': 'capital_gain',\
                            ' capital-loss': 'capital_loss',\
                            ' hours-per-week': 'hrs_per_week',\
                            ' native-country': 'native_country',\
                            ' income': 'income'})

    return d



