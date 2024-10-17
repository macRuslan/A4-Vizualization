# Import python packages
import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Custom CSS for black background and white text
st.markdown("""
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stMarkdown h2, .stMarkdown h1, .stMarkdown p {
        color: white;
    }
    .stCheckbox {
        color: white;
    }
    .css-1d391kg, .css-1v3fvcr {
        background-color: black;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('kaggle.csv')
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

# --- New Chart 1: Age vs. Stroke Risk by Gender ---
st.subheader('Stroke Risk by Age and Gender')

# Create a bar plot for stroke risk by age and gender
age_gender_fig = px.histogram(data, x='age', color='gender', barmode='group',
                              labels={'age': 'Age', 'gender': 'Gender'},
                              hover_data=['stroke', 'smoking_status', 'bmi'])
st.plotly_chart(age_gender_fig)

# --- Updated Chart 2: Lifestyle Factors without Facet ---
st.subheader('Lifestyle Factors Impacting Stroke Risk')

# Create scatter plot for BMI vs. glucose level, colored by stroke risk
lifestyle_fig = px.scatter(data, x='bmi', y='avg_glucose_level', color='stroke',
                           labels={'bmi': 'BMI', 'avg_glucose_level': 'Average Glucose Level', 'stroke': 'Stroke'},
                           hover_data=['smoking_status', 'hypertension', 'age'])
st.plotly_chart(lifestyle_fig)

# --- New Chart 3: Stroke Risk for Older Men vs. Women with Lifestyle Factors ---
st.subheader('Stroke Risk for Older Men vs. Women by Lifestyle Factors')

# Filter data for older men and women (55 and above)
older_data = data[data['age'] >= 55]

# Create a box plot showing stroke risk based on smoking status, BMI, and glucose levels, for men and women
boxplot_fig = px.box(older_data, x='gender', y='bmi', color='stroke', facet_row='smoking_status',
                     labels={'bmi': 'BMI', 'smoking_status': 'Smoking Status', 'stroke': 'Stroke'},
                     hover_data=['avg_glucose_level', 'hypertension'])
st.plotly_chart(boxplot_fig)

# --- New Chart 4: Age-Specific Stroke Predictors (Heatmap) ---
st.subheader('Stroke Predictors by Age Groups (Under 55 vs. Over 55)')

# Split data into under 55 and over 55 groups for comparison
data_under_55 = data[data['age'] < 55]
data_over_55 = data[data['age'] >= 55]

# Correlation heatmap for individuals under 55
st.subheader('Correlation for Individuals Under 55')
corr_under_55 = data_under_55[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
fig_under_55 = px.imshow(corr_under_55, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
st.plotly_chart(fig_under_55)

# Correlation heatmap for individuals over 55
st.subheader('Correlation for Individuals Over 55')
corr_over_55 = data_over_55[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
fig_over_55 = px.imshow(corr_over_55, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
st.plotly_chart(fig_over_55)

# Encode Categorical Variables
data_encoded = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# Feature Selection
features = [col for col in data_encoded.columns if col not in ['id', 'stroke']]
X = data_encoded[features]
y = data_encoded['stroke']

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Model Training - Logistic Regression
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

# Create a Matplotlib figure for SHAP plot with dark theme
fig, ax = plt.subplots(facecolor='black')
shap.plots.bar(shap_values, max_display=10, show=False, ax=ax)
ax.set_facecolor('black')  # Make the background black
ax.tick_params(colors='white')  # Make the ticks white
ax.yaxis.label.set_color('white')  # Set y-axis label color to white
ax.xaxis.label.set_color('white')  # Set x-axis label color to white
ax.title.set_color('white')  # Set title color to white
st.pyplot(fig)

# Alternative Model: Random Forest
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