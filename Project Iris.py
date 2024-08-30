import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


file_path = "F:\proper python\encoded-IRIS.csv"
df = pd.read_csv(file_path)

#desriptive analysis
print(df.info())
print(df.describe())
print(df.isnull().sum())

#encoding the categorical variable(Species)
label_encoder = LabelEncoder()
df['species_encoded'] = label_encoder.fit_transform(df['species'])

#defining X & Y
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species_encoded']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


#setting the model up
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

#performance metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)


#plotting the confusion matrix 
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens', ax=ax, 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
ax.set_title('Random Forest - Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()

#storing the results in a single table 
results = {
    "Metric": ["Accuracy", "Precision (Class 0)", "Precision (Class 1)", "Precision (Class 2)", 
               "Recall (Class 0)", "Recall (Class 1)", "Recall (Class 2)", 
               "F1-Score (Class 0)", "F1-Score (Class 1)", "F1-Score (Class 2)"],
    "Random Forest": [
        accuracy_rf,
        class_report_rf['0']['precision'],
        class_report_rf['1']['precision'],
        class_report_rf['2']['precision'],
        class_report_rf['0']['recall'],
        class_report_rf['1']['recall'],
        class_report_rf['2']['recall'],+
        class_report_rf['0']['f1-score'],
        class_report_rf['1']['f1-score'],
        class_report_rf['2']['f1-score']
    ]
}

results_df = pd.DataFrame(results)
print(results_df)