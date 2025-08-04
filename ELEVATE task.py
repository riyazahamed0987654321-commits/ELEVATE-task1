import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Titanic Dataset
df = pd.read_csv('titanic.csv')

#Explore basic info
print(df.head())
print(df.info())
print(df.describe())

#Check for missing values
print(df.isnull().sum())

#Handle missing values in the 'Age' column using mean imputation
df['Age'].fillna(df['Age'].mean(), inplace=True)

#Handle missing values in the 'Cabin' column by dropping it
df.drop('Cabin', axis=1, inplace=True)

#Convert categorical features into numerical ones using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

#Normalize numerical features using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

#Visualize outliers using boxplots
sns.boxplot(x=df['Age'])
plt.show()

#Remove outliers using IQR method
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR)))]
