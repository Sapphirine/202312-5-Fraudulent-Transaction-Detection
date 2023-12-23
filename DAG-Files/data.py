# Libraries
import numpy as np
import pandas as pd
import os
import gc
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
# ML packages
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Loading data
folder_path = '/home/ggvkulkarni'
train_identity = pd.read_csv(f'{folder_path}/train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}/train_transaction_(1).csv')
sub = pd.read_csv(f'{folder_path}/sample_submission.csv')
# let's combine the data and work with the whole dataset
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

# pre-processing
del train_identity, train_transaction

one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

# Feature Engineering
train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')

train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)
train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)

# Data Pre-processing
train["isFraud"].value_counts()

many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.7]

big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.7]

cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
cols_to_drop.remove('isFraud')

train = train.drop(cols_to_drop, axis=1)

cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
for col in cat_cols:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))

X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
del train

def clean_inf_nan_chunked(df, chunksize=10000):
    total_rows = len(df)
    chunks = [df[i:i + chunksize] for i in range(0, total_rows, chunksize)]

    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}")
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Concatenate the processed chunks back into a DataFrame
    return pd.concat(chunks, ignore_index=True)

# Clean infinite values to NaN in chunks
X = clean_inf_nan_chunked(X)

# Oversampling
undersample = RandomUnderSampler(sampling_strategy=0.225)
# oversample = RandomOverSampler(sampling_strategy=0.6)
X_over, y_over = undersample.fit_resample(X, y)

# Concatenate X and Y for simplicity
df = pd.concat([X_over, y_over], axis=1)


df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cleaned = df.interpolate(method='linear')


imputer = SimpleImputer(strategy='mean')  # You can choose 'median' or 'constant' as well
df_cleaned = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Separate X and Y after cleaning
X_cleaned = df_cleaned[X_over.columns]
Y_cleaned = df_cleaned['isFraud']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, Y_cleaned, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#pca = PCA(n_components=50)

#X_train_scaled = pca.fit_transform(X_train_scaled)
#X_test_scaled = pca.transform(X_test_scaled)

PATH = '/home/ggvkulkarni/proj/data/'

X_train.to_csv(PATH + 'xtrain.csv', index = False)
X_test.to_csv(PATH + 'xtest.csv', index = False)
y_train.to_csv(PATH + 'ytrain.csv', index = False)
y_test.to_csv(PATH + 'ytest.csv', index = False)
