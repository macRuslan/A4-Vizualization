import streamlit as st
import pandas as pd
import plotly.express as px
import shap
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
    gender = st.sidebar.multiselect('Select Gender', options=data['gender'].unique(), default=data['gender'].unique())
    hypertension = st.sidebar.checkbox('Include Only Hypertensive Patients', value=False)
    return age_group, gender, hypertension

age_group, gender, hypertension = user_input_features()

# Filter data based on user input
if age_group == 'Under 55':
    data = data[data['age'] < 55]
elif age_group == '55 and above':
    data = data[data['age'] >= 55]

data = data[data['gender'].isin(gender)]

if hypertension:
    data = data[data['hypertension'] == 1]

# Display dataset
if st.checkbox('Show Dataset'):
    st.write(data)

# Interactive Correlation Heatmap
st.subheader('Interactive Correlation Heatmap')
corr = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
st.plotly_chart(fig)

# Stroke Distribution by Smoking Status
st.subheader('Stroke Distribution by Smoking Status')
fig = px.histogram(data, x='smoking_status', color='stroke', barmode='group',
                   category_orders={'smoking_status': ['never smoked', 'formerly smoked', 'smokes', 'Unknown']},
                   labels={'stroke': 'Stroke', 'smoking_status': 'Smoking Status'})
st.plotly_chart(fig)

# BMI vs. Average Glucose Level Scatter Plot
st.subheader('BMI vs. Average Glucose Level')
fig = px.scatter(data, x='bmi', y='avg_glucose_level', color='stroke',
                 labels={'bmi': 'BMI', 'avg_glucose_level': 'Average Glucose Level', 'stroke': 'Stroke'},
                 hover_data=['age', 'hypertension', 'heart_disease'])
st.plotly_chart(fig)

# Age Distribution with Stroke Outcome
st.subheader('Age Distribution with Stroke Outcome')
fig = px.histogram(data, x='age', color='stroke', nbins=50, opacity=0.7,
                   labels={'age': 'Age', 'stroke': 'Stroke'})
st.plotly_chart(fig)

# Encode Categorical Variables
data_encoded = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# Feature Selection
features = [col for col in data_encoded.columns if col not in ['id', 'stroke']]
X = data_encoded[features]
y = data_encoded['stroke']

# Train-test Split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handling class imbalance with SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Model Training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# Evaluation: Classification Report
st.write('**Logistic Regression Classification Report:**')
st.text(classification_report(y_test, y_pred, zero_division=0))

# Check if both classes are present in y_test before calculating ROC AUC score
if len(set(y_test)) > 1:
    st.write('**Logistic Regression ROC AUC Score:**', roc_auc_score(y_test, y_pred_proba))
else:
    st.write('**ROC AUC Score cannot be calculated:** Only one class present in the selected data.')

# Feature Importance with SHAP
st.subheader('Feature Importance with SHAP Values')

# Create a SHAP explainer and calculate SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Create a Matplotlib figure for SHAP plot
fig, ax = plt.subplots()

# Plot SHAP values
shap.plots.bar(shap_values, max_display=10, show=False, ax=ax)

# Display the figure in Streamlit
st.pyplot(fig)

# Alternative Model: Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Predictions
rf_y_pred = rf_model.predict(X_test)
rf_y_pred_proba = rf_model.predict_proba(X_test)[:,1]

# Evaluation
st.write('**Random Forest Classification Report:**')
st.text(classification_report(y_test, rf_y_pred, zero_division=0))

# Check if both classes are present in y_test before calculating ROC AUC score for Random Forest
if len(set(y_test)) > 1:
    st.write('**Random Forest ROC AUC Score:**', roc_auc_score(y_test, rf_y_pred_proba))
else:
    st.write('**ROC AUC Score cannot be calculated for Random Forest:** Only one class present in the selected data.')

# Model Comparison
st.subheader('Model Comparison')
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'ROC AUC Score': [roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 'N/A', 
                      roc_auc_score(y_test, rf_y_pred_proba) if len(set(y_test)) > 1 else 'N/A']
})
st.write(models)