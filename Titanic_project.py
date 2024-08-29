import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import statsmodels.api as sm
from scipy.stats import chi2

file_path = 'F:\\proper python\\encoded-Titanic-Dataset.csv'
df = pd.read_csv(file_path)

#some descriptive analysis of the data
print(df.info())
print(df.describe())
print(df.isnull().sum())



#data cleaning and variable creation
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
df['family_size'] = df['SibSp'] + df['Parch']
df.drop(['SibSp', 'Parch'], axis=1, inplace=True) 
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
imputer = SimpleImputer(strategy='median')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Create dummy variables for Embarked ports
df['Cherbourg'] = df['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
df['Queenstown'] = df['Embarked'].apply(lambda x: 1 if x == 'Q' else 0)
df.drop('Embarked', axis=1, inplace=True)

# Calculating survival rate by sex and plotting it in a bar graph 
survival_rate_by_sex = df.groupby('Sex')['Survived'].mean().reset_index()
survival_rate_by_sex.columns = ['Sex', 'Survival Rate']
plt.figure(figsize=(8, 6))
sns.barplot(x='Sex', y='Survival Rate', data=survival_rate_by_sex, palette='viridis')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
plt.ylim(0, 1)  
plt.show()


# Calculating survival rate by cabin class and plotting it ina bar graph 
survival_rate_by_class = df.groupby('Pclass')['Survived'].mean().reset_index()
survival_rate_by_class.columns = ['Pclass', 'Survival Rate']
plt.figure(figsize=(8, 6))
sns.barplot(x='Pclass', y='Survival Rate', data=survival_rate_by_class, palette='viridis')
plt.title('Survival Rate by Cabin Class')
plt.xlabel('Cabin Class (Pclass)')
plt.ylabel('Survival Rate')
plt.ylim(0, 1) 
plt.show()


# define X & Y
X = df[['Fare', 'Sex', 'Pclass', 'family_size', 'Age', 'Cherbourg', 'Queenstown']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add the intercept 
X_train_sm = sm.add_constant(X_train)


#set up the logistic regression 
logistic_reg = sm.Logit(y_train, X_train_sm).fit()
print(logistic_reg.summary())

#  Hosmer-Lemeshow Test 
def hosmer_lemeshow_test(y_true, y_prob, g=10):
   
    data = pd.DataFrame({'true': y_true, 'prob': y_prob})
    data['group'] = pd.qcut(data['prob'], g, duplicates='drop')
    observed = data.groupby('group')['true'].sum()
    expected = data.groupby('group')['prob'].mean() * data.groupby('group')['true'].count()
    hl_stat = (((observed - expected) ** 2) / (expected * (1 - expected / data.groupby('group')['true'].count()))).sum()
    p_value = 1 - chi2.cdf(hl_stat, g - 2)
    return hl_stat, p_value
y_train_prob = logistic_reg.predict(X_train_sm)
hl_stat, hl_p_value = hosmer_lemeshow_test(y_train, y_train_prob)
print(f"Hosmer-Lemeshow Test Statistic: {hl_stat:.4f}, p-value: {hl_p_value:.4f}")

#logit,randomforest,decisiontree

#logit
logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train, y_train)
y_pred_logreg = logistic_reg.predict(X_test)

#randomforest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#decision tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

#performance comparison 
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

class_report_logitreg = classification_report(y_test, y_pred_logreg, output_dict=True)
class_report_randomforest = classification_report(y_test, y_pred_rf, output_dict=True)
class_report_dtree = classification_report(y_test, y_pred_dt, output_dict=True)

# Confusion matrices
conf_matrix_logitreg = confusion_matrix(y_test, y_pred_logreg)
conf_matrix_randomforest = confusion_matrix(y_test, y_pred_rf)
conf_matrix_dtree = confusion_matrix(y_test, y_pred_dt)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(conf_matrix_logitreg, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_randomforest, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

sns.heatmap(conf_matrix_dtree, annot=True, fmt='d', cmap='Reds', ax=axes[2])
axes[2].set_title('Decision Tree')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.show()

results = {
    "Metric": ["Accuracy", "Precision (Class 0)", "Precision (Class 1)", "Recall (Class 0)", "Recall (Class 1)", "F1-Score (Class 0)", "F1-Score (Class 1)"],
    "Logistic Regression": [
        accuracy_logreg,
        class_report_logitreg['0']['precision'],
        class_report_logitreg['1']['precision'],
        class_report_logitreg['0']['recall'],
        class_report_logitreg['1']['recall'],
        class_report_logitreg['0']['f1-score'],
        class_report_logitreg['1']['f1-score']
    ],
    "Random Forest": [
        accuracy_rf,
        class_report_randomforest['0']['precision'],
        class_report_randomforest['1']['precision'],
        class_report_randomforest['0']['recall'],
        class_report_randomforest['1']['recall'],
        class_report_randomforest['0']['f1-score'],
        class_report_randomforest['1']['f1-score']
    ],
    "Decision Tree": [
        accuracy_dt,
        class_report_dtree['0']['precision'],
        class_report_dtree['1']['precision'],
        class_report_dtree['0']['recall'],
        class_report_dtree['1']['recall'],
        class_report_dtree['0']['f1-score'],
        class_report_dtree['1']['f1-score']
    ]
}

results_df = pd.DataFrame(results)
print(results_df)
