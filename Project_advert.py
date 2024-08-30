import pandas as pd
import statsmodels.api as sm


file_name = "F:\proper python\encoded-advertising.csv"
df= pd.read_csv(file_name)

#data into and descriptive analysis 
print(df.info())
print(df.describe())
print(df.isnull().sum())

#no data cleaning needed as the dataset was perfectly balanced

#defining X & Y
X = df[['TV', 'Radio', 'Newspaper']]
Y= df['Sales']

X = sm.add_constant(X)

#setting up the model
model = sm.OLS(Y, X)
results = model.fit()

#results
print(results.summary())