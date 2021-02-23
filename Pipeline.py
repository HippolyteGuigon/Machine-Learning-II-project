from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

le = preprocessing.LabelEncoder()

def clean_data(X):
    le = LabelEncoder()
    categorical = []
    error = []
    missing = X[X.isna().sum()[X.isna().sum()>0].index].dtypes[X[X.isna().sum()[X.isna().sum()>0].index].dtypes=="float64"].index
    for col in missing:
        X[col] = X[col].fillna(np.mean(X[col]))
    for i in X.columns:
        if X[i].dtypes=="O":
            categorical.append(i)
    for i in range(0, len(categorical)):
        if X[categorical[i]].isna().sum() == 0:
            X[categorical[i]] = le.fit_transform(X[categorical[i]])
        else:
            error.append(categorical[i])
    X[error] = X[error].fillna("No such thing")
    for i in range(0, len(error)):
        X[error][i] = le.fit_transform(X[error][i])
    return X

def graphique_time(X):
    names = pd.get_dummies(X["type"]).columns
    values = [pd.get_dummies(X["type"]).iloc[:, i].sum() for i in range(len(pd.get_dummies(X["type"]).columns))]
    second_time = [X[X["type"]==i]["NbSecondWithAtLeatOneTrade"].mean() for i in list(pd.get_dummies(X["type"]).columns)]

    fig, ax1 = plt.subplots(figsize=(15, 10))
    plt.title("Average second/Type of Trader", fontsize=15)
    s1 = values
    ax1.bar(names, values)
    ax1.set_xlabel('Type of trader', fontsize=15)
    ax1.set_ylabel('Number of trader', color='b', fontsize=15);
    ax2 = ax1.twinx()
    s2 = second_time
    ax2.plot(second_time, 'o', color='red')
    ax2.annotate("480.7", xy=(200, 545), xycoords='figure pixels', color="red")
    ax2.annotate("364.8", xy=(468, 415), xycoords='figure pixels', color="red")
    ax2.annotate("53.5", xy=(745, 53), xycoords='figure pixels', color="red")
    ax2.set_ylabel('Average second', color='r', fontsize=15);
    
def graphique_normal(X):
    fig = plt.figure(figsize=(20, 15))
    sns.set(font_scale=1.5)

    # (Corr= 0.817185) Box plot overallqual/salePrice
    fig4 = fig.add_subplot(221);
    sns.boxplot(x='type', y='NbTradeVenueMic', data=X[['NbTradeVenueMic', 'type']])

    # (Corr= 0.700927) GrLivArea vs SalePrice plot
    fig2 = fig.add_subplot(222); 
    sns.scatterplot(x = X[X["OTR"]<=5000]["OTR"], y = X[X["NbSecondWithAtLeatOneTrade"]<=9000]["NbSecondWithAtLeatOneTrade"], hue=X.type, palette= 'Spectral')

    # (Corr= 0.700927) GrLivArea vs SalePrice plot
    fig3 = fig.add_subplot(223); 
    sns.scatterplot(x = X["NbTradeVenueMic"], y = X["NbSecondWithAtLeatOneTrade"], hue=X.type, palette= 'Spectral')

    plt.tight_layout(); plt.show()
    
    
