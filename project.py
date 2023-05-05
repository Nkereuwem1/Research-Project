# Prediction of Stimulant Use Disorder

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pickle
import sklearn
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

drugs = pd.read_csv(r"C:\Users\peezed\Desktop\Nkereuwem\Dataset\Drug_Consumption.csv")
drugs.head()

drugs = drugs.drop("ID", axis=1)
drugs.head()

# check for missing values
drugs.isna().sum().sum()

print(f"original shape of data with {drugs.shape[0]} rows and {drugs.shape[1]} columns")

# overclaimers
drugs.query("Semer !='CL0'")

drugs = drugs.drop(drugs[drugs['Semer'] !='CL0'].index)

drugs = drugs.drop(['Choc', 'Semer'], axis=1)
drugs = drugs.reset_index(drop=True)
drugs.head()

# Feature Encoding

stimulants = ['Alcohol', 'Amyl', 'Amphet', 'Benzos', 'Caff', 'Cannabis', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']

def stimulant_encoder(x):
    if x == 'CL0':
        return 0
    elif x == 'CL1':
        return 1
    elif x == 'CL2':
        return 2
    elif x == 'CL3':
        return 3
    elif x == 'CL4':
        return 4
    elif x == 'CL5':
        return 5
    elif x == 'CL6':
        return 6
    else:
        return 7

for column in stimulants:
    drugs[column] = drugs[column].apply(stimulant_encoder)

drugs.head()

corr = drugs.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, vmin=-1)

low_corr = ['Alcohol', 'AScore', 'Caff']
for column in low_corr:
    drugs = drugs.drop(column, axis=1)
drugs.head()

print(f'In the new dataframe there are {drugs.shape[0]} rows and {drugs.shape[1]} columns')

# FEATURE ENGINEERING

# Combine cocaine and crack cocaine usage into one feature
cocaine_df = drugs.copy()
cocaine_df['coke_user'] = cocaine_df['Coke'].apply(lambda x: 0.5 if x not in [0,1] else 0)
cocaine_df['crack_user'] = cocaine_df['Coke'].apply(lambda x: 0.5 if x not in [0,1] else 0)
cocaine_df['both_user'] = cocaine_df[['coke_user', 'crack_user']].iloc[:].sum(axis=1)
cocaine_df['Cocaine_User'] = cocaine_df['both_user'].apply(lambda x: 1 if x > 0 else 0)
cocaine_df = cocaine_df.drop(['coke_user', 'crack_user', 'both_user' ], axis=1)

meth_df = drugs.copy()
meth_df['Meth_User'] = meth_df['Meth'].apply(lambda x: 1 if x not in [0,1] else 0)
meth_df = meth_df.drop(['Meth'], axis=1)

heroin_df = drugs.copy()
heroin_df['Heroin_User'] = heroin_df['Heroin'].apply(lambda x: 1 if x not in [0,1] else 0)
heroin_df = heroin_df.drop(['Heroin'], axis=1)

nic_df = drugs.copy()
nic_df['Nicotine_User'] = nic_df['Nicotine'].apply(lambda x: 1 if x not in [0,1] else 0)
nic_df = nic_df.drop(['Nicotine'], axis=1)

cocaine_df.head(1)

meth_df.head(1)

heroin_df.head(1)

nic_df.head(1)

# DATA PREPROCESSING

def preprocessing_inputs(df, column):
    df = df.copy()
    
    # Split df into X and y
    y = df[column]
    X = df.drop(column, axis=1)
    
     # Add missing features to X
    X["Age"] = df["Age"]
    X["Education"] = df["Education"]
    X["Gender"] = df["Gender"]
    
    # Convert categorical columns to numerical
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = LabelEncoder()
        X[categorical_cols] = X[categorical_cols].apply(encoder.fit_transform)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = pd.DataFrame(scaler.transform(X_train), 
                           index=X_train.index, 
                           columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), 
                          index=X_test.index, 
                          columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y,y_predict):
    #Function to easily plot confusion matrix
    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Blues');
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['non-user', 'user']); ax.yaxis.set_ticklabels(['non-user', 'user'])

# Cocaine

# Model Training

X_train, X_test, y_train, y_test = preprocessing_inputs(cocaine_df, 'Cocaine_User')

X_train.head()

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

models = {
            '     Logistic Regression': LogisticRegression(),
            ' Support Vector Machines': SVC(), 'Random Forest Classifier': RandomForestClassifier()}

for name, model_cocaine in models.items():
    model_cocaine.fit(X_train, y_train)
    print(name + ' trained.')

# Model Results

print('                  ACCURACY')
for name, model_cocaine in models.items():
    yhat = model_cocaine.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    
    print(name + ' Accuracy: {:.2%}'.format(acc))
print('---------------------------------------------')
print('                  F1 SCORES')
for name, model_cocaine in models.items():
    yhat = model_cocaine.predict(X_test)
    f1 = f1_score(y_test, yhat, pos_label=1)
    print(name + ' F1-Score: {:.5}'.format(f1))

#Confusion Matrix of Best Peforming Model
model_cocaine = LogisticRegression()
model_cocaine.fit(X_train, y_train)
yhat = model_cocaine.predict(X_test)
plot_confusion_matrix(y_test, yhat)

with open("cocaine_model", 'wb') as f:
    pickle.dump(model_cocaine, f)

print(model_cocaine.feature_names_in_)

print(drugs.columns)

# Methamphetamine

# Model Training

X_train, X_test, y_train, y_test = preprocessing_inputs(meth_df, 'Meth_User')

for name, model_meth in models.items():
    model_meth.fit(X_train, y_train)
    print(name + ' trained.')

# Model Results

print('                  ACCURACY')
for name, model_meth in models.items():
    yhat = model_meth.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    print(name + ' Accuracy: {:.2%}'.format(acc))
print('---------------------------------------------')
print('                  F1 SCORES')
for name, model_meth in models.items():
    yhat = model_meth.predict(X_test)
    f1 = f1_score(y_test, yhat, pos_label=1)
    print(name + ' F1-Score: {:.5}'.format(f1))

model_meth = RandomForestClassifier()
model_meth.fit(X_train, y_train)
yhat = model_meth.predict(X_test)
plot_confusion_matrix(y_test, yhat)

with open("meth_model", 'wb') as f:
    pickle.dump(model_meth, f)

# Heroin 

# Model Training

X_train, X_test, y_train, y_test = preprocessing_inputs(heroin_df, 'Heroin_User')

print(heroin_df['Heroin_User'].unique())

for name, model_heroin in models.items():
    model_heroin.fit(X_train, y_train)
    print(name + ' trained.')

# Model Results

print('                  ACCURACY')
for name, model_heroin in models.items():
    yhat = model_heroin.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    print(name + ' Accuracy: {:.2%}'.format(acc))
print('---------------------------------------------')
print('                  F1 SCORES')
for name, model_heroin in models.items():
    yhat = model_heroin.predict(X_test)
    f1 = f1_score(y_test, yhat, pos_label=1)
    print(name + ' F1-Score: {:.5}'.format(f1))

model_heroin = SVC()
model_heroin.fit(X_train, y_train)
yhat = model_heroin.predict(X_test)
plot_confusion_matrix(y_test, yhat)

with open("heroin_model", 'wb') as f:
    pickle.dump(model_heroin, f)

# Nicotine

# Model Training

X_train, X_test, y_train, y_test = preprocessing_inputs(nic_df, 'Nicotine_User')

for name, model_nico in models.items():
    model_nico.fit(X_train, y_train)
    print(name + ' trained.')

print('                  ACCURACY')
for name, model_nico in models.items():
    yhat = model_nico.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    print(name + ' Accuracy: {:.2%}'.format(acc))
print('---------------------------------------------')
print('                  F1 SCORES')
for name, model_nico in models.items():
    yhat = model_nico.predict(X_test)
    f1 = f1_score(y_test, yhat, pos_label=1)
    print(name + ' F1-Score: {:.5}'.format(f1))

model_nico = SVC()
model_nico.fit(X_train, y_train)
yhat = model_nico.predict(X_test)
plot_confusion_matrix(y_test, yhat)

with open("nico_model", 'wb') as f:
    pickle.dump(model_nico, f)

