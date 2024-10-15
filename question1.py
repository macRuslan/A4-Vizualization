import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('/Users/ruslankhissamiyev/Desktop/BUSA 8031 A4/A4-Vizualization/Datasets/kaggle.csv')
    return data

data = load_data()

# Preprocess the data
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

# Title and Description
st.title('Stroke Risk Factors Analysis in NSW')
st.markdown("""
This dashboard analyzes significant predictors of stroke among different age groups in NSW, focusing on lifestyle factors such as smoking status, BMI, and average glucose levels.
""")

# Sidebar for User Input
st.sidebar.header('User Input Features')

# Function to get user input
def user_input_features():
    age_group = st.sidebar.selectbox('Select Age Group', ('All', 'Under 55', '55 and above'))
    return age_group

age_group = user_input_features()

# Filter data based on age group
if age_group == 'Under 55':
    data = data[data['age'] < 55]
elif age_group == '55 and above':
    data = data[data['age'] >= 55]

# Display dataset
if st.checkbox('Show Dataset'):
    st.write(data)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
corr = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Stroke Distribution by Smoking Status
st.subheader('Stroke Distribution by Smoking Status')
fig, ax = plt.subplots()
sns.countplot(x='smoking_status', hue='stroke', data=data, ax=ax)
ax.set_title('Stroke Distribution by Smoking Status')
st.pyplot(fig)

# BMI vs. Average Glucose Level Scatter Plot
st.subheader('BMI vs. Average Glucose Level')
fig, ax = plt.subplots()
scatter = ax.scatter(data['bmi'], data['avg_glucose_level'], c=data['stroke'], cmap='bwr', alpha=0.7)
ax.set_xlabel('BMI')
ax.set_ylabel('Average Glucose Level')
ax.set_title('BMI vs. Average Glucose Level Colored by Stroke Outcome')
legend1 = ax.legend(*scatter.legend_elements(), title="Stroke")
ax.add_artist(legend1)
st.pyplot(fig)

# Age Distribution with Stroke
st.subheader('Age Distribution with Stroke Outcome')
fig, ax = plt.subplots()
sns.histplot(data, x='age', hue='stroke', multiple='stack', ax=ax)
ax.set_title('Age Distribution with Stroke Outcome')
st.pyplot(fig)

# Logistic Regression Model
st.subheader('Predictive Model: Logistic Regression')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Feature Selection
features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
X = data[features]
y = data['stroke']

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# Evaluation
st.write('**Classification Report:**')
st.text(classification_report(y_test, y_pred, zero_division=0))

st.write('**ROC AUC Score:**', roc_auc_score(y_test, y_pred_proba))

# Feature Importance
st.subheader('Feature Importance')
importance = pd.Series(model.coef_[0], index=features)
fig, ax = plt.subplots()
importance.plot.bar(ax=ax)
ax.set_title('Feature Importance in Predicting Stroke')
st.pyplot(fig)