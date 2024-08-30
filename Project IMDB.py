import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

file_path = "F:\\proper python\\encoded-IMDb Movies India.csv"
df = pd.read_csv(file_path)

print(df.info())
print(df.describe())
print(df.isnull().sum())


# Data Cleaning
df.dropna(inplace=True)
df['Year'] = df['Year'].str.extract(r'(\d+)').astype(float)  
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)  
df['Votes'] = df['Votes'].str.replace(',', '').astype(int) 
df["Directors"] = df['Director'].astype('category').cat.codes
df["Genres"] = df['Genre'].astype('category').cat.codes


rating_variance = df['Rating'].var()

plt.figure(figsize=(10, 6))
plt.hist(df['Rating'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(df['Rating'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {df['Rating'].mean():.2f}")
plt.axvline(df['Rating'].mean() + rating_variance**0.5, color='green', linestyle='dotted', linewidth=2, label=f"+1 SD: {(df['Rating'].mean() + rating_variance**0.5):.2f}")
plt.axvline(df['Rating'].mean() - rating_variance**0.5, color='green', linestyle='dotted', linewidth=2, label=f"-1 SD: {(df['Rating'].mean() - rating_variance**0.5):.2f}")

plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()



# Defining X and Y
X = df.drop(['Name', 'Genre', 'Rating', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)
Y = df['Rating']

# Splitting the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Evaluation metrics
def evaluate_model(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Model: {model_name}")
    print(f"r2 (R^2) = {r2:.2f}")
    print(f"Root Mean Squared Error = {mse:.2f}")
    print(f"Mean Absolute Error = {mae:.2f}")
    
    return r2, mse, mae



# Linear regression 
LR = LinearRegression()
LR.fit(x_train, y_train)
lr_preds = LR.predict(x_test)
LRScore, LR_mse, LR_MAE = evaluate_model(y_test, lr_preds, "LINEAR REGRESSION")

# Random Forest Regressor
RFR = RandomForestRegressor(n_estimators=100, random_state=1)
RFR.fit(x_train, y_train)
rf_preds = RFR.predict(x_test)
RFScore, RF_mse, RF_MAE = evaluate_model(y_test, rf_preds, "RANDOM FOREST")

# Storing the results and sorting them via r2 score
models = pd.DataFrame(
    {
        "MODELS": ["Linear Regression", "Random Forest"], 
        "R^2 SCORE": [LRScore, RFScore],
        "mse": [LR_mse, RF_mse],
        "MAE": [LR_MAE, RF_MAE]
    }
)
models.sort_values(by='R^2 SCORE', ascending=False, inplace=True)
print(models)
