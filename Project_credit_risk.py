import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

file_name = "F:\proper python\encoded-creditcard.csv"
df = pd.read_csv(file_name)

print(df.info())
print(df.describe())
print(df.isnull().sum())

#normalizing the data 
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df[['Time']].values.reshape(-1, 1))

# Define X & Y
X = df.drop('Class', axis=1)
y = df['Class']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#SMOTE ((Synthetic Minority Oversampling Technique) used to account for the variance in the dataset)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print('After SMOTE:')
print('X_train_sm shape:', X_train_sm.shape)
print('y_train_sm shape:', y_train_sm.shape)


# Random Forest Model 
print("Fitting the model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_sm, y_train_sm)

# Predict probabilities to calculate precision-Recall and AUPRC
print("Predicting...")
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Calculation of Precision-Recall and AUPRC
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
auprc = auc(recall, precision)
print(f'AUPRC: {auprc:.2f}')

y_pred = model.predict(X_test)
print('Classification Report:\n', classification_report(y_test, y_pred))
