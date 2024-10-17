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

# --- Original Visualization: Correlation Heatmap ---
st.subheader('Original: Correlation Heatmap')
corr = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
st.plotly_chart(fig)

# --- Original Visualization: Stroke Distribution by Smoking Status ---
st.subheader('Original: Stroke Distribution by Smoking Status')
fig = px.histogram(data, x='smoking_status', color='stroke', barmode='group',
                   category_orders={'smoking_status': ['never smoked', 'formerly smoked', 'smokes', 'Unknown']},
                   labels={'stroke': 'Stroke', 'smoking_status': 'Smoking Status'},
                   hover_data=['age', 'hypertension', 'heart_disease'])
st.plotly_chart(fig)

# --- Original Visualization: BMI vs. Average Glucose Level Scatter Plot ---
st.subheader('Original: BMI vs. Average Glucose Level')
fig = px.scatter(data, x='bmi', y='avg_glucose_level', color='stroke',
                 labels={'bmi': 'BMI', 'avg_glucose_level': 'Average Glucose Level', 'stroke': 'Stroke'},
                 hover_data=['age', 'hypertension', 'heart_disease'])
st.plotly_chart(fig)

# --- Original Visualization: Age Distribution with Stroke Outcome ---
st.subheader('Original: Age Distribution with Stroke Outcome')
fig = px.histogram(data, x='age', color='stroke', nbins=50, opacity=0.7,
                   labels={'age': 'Age', 'stroke': 'Stroke'},
                   hover_data=['bmi', 'avg_glucose_level'])
st.plotly_chart(fig)

# --- New Visualization 1: Age vs. Stroke Risk by Gender ---
st.subheader('New: Stroke Risk by Age and Gender')
age_gender_fig = px.histogram(data, x='age', color='gender', barmode='group',
                              labels={'age': 'Age', 'gender': 'Gender'},
                              hover_data=['stroke', 'smoking_status', 'bmi'])
st.plotly_chart(age_gender_fig)

# --- New Visualization 2: Lifestyle Factors without Facet ---
st.subheader('New: Lifestyle Factors Impacting Stroke Risk')
lifestyle_fig = px.scatter(data, x='bmi', y='avg_glucose_level', color='stroke',
                           labels={'bmi': 'BMI', 'avg_glucose_level': 'Average Glucose Level', 'stroke': 'Stroke'},
                           hover_data=['smoking_status', 'hypertension', 'age'])
st.plotly_chart(lifestyle_fig)

# --- New Visualization 3: Stroke Risk for Older Men vs. Women with Lifestyle Factors ---
st.subheader('New: Stroke Risk for Older Men vs. Women by Lifestyle Factors')
older_data = data[data['age'] >= 55]
boxplot_fig = px.box(older_data, x='gender', y='bmi', color='stroke', facet_row='smoking_status',
                     labels={'bmi': 'BMI', 'smoking_status': 'Smoking Status', 'stroke': 'Stroke'},
                     hover_data=['avg_glucose_level', 'hypertension'])
st.plotly_chart(boxplot_fig)

# --- New Visualization 4: Age-Specific Stroke Predictors (Heatmap) ---
st.subheader('New: Stroke Predictors by Age Groups (Under 55 vs. Over 55)')
data_under_55 = data[data['age'] < 55]
data_over_55 = data[data['age'] >= 55]
st.subheader('Correlation for Individuals Under 55')
corr_under_55 = data_under_55[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
fig_under_55 = px.imshow(corr_under_55, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
st.plotly_chart(fig_under_55)
st.subheader('Correlation for Individuals Over 55')
corr_over_55 = data_over_55[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].corr()
fig_over_55 = px.imshow(corr_over_55, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
st.plotly_chart(fig_over_55)

# --- Model Training ---
data_encoded = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
features = [col for col in data_encoded.columns if col not in ['id', 'stroke']]
X = data_encoded[features]
y = data_encoded['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

st.write('**Logistic Regression Classification Report:**')
st.text(classification_report(y_test, y_pred, zero_division=0))

if len(set(y_test)) > 1:
    st.write('**Logistic Regression ROC AUC Score:**', roc_auc_score(y_test, y_pred_proba))
else:
    st.write('**ROC AUC Score cannot be calculated:** Only one class present in the selected data.')

st.subheader('Feature Importance with SHAP Values')
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
fig, ax = plt.subplots(facecolor='black')
shap.plots.bar(shap_values, max_display=10, show=False, ax=ax)
ax.set_facecolor('black')
ax.tick_params(colors='white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.title.set_color('white')
st.pyplot(fig)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)
rf_y_pred = rf_model.predict(X_test)
rf_y_pred_proba = rf_model.predict_proba(X_test)[:,1]

st.write('**Random Forest Classification Report:**')
st.text(classification_report(y_test, rf_y_pred, zero_division=0))

if len(set(y_test)) > 1:
    st.write('**Random Forest ROC AUC Score:**', roc_auc_score(y_test, rf_y_pred_proba))
else:
    st.write('**ROC AUC Score cannot be calculated for Random Forest:** Only one class present in the selected data.')

st.subheader('Model Comparison')
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'ROC AUC Score': [roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 'N/A', 
                      roc_auc_score(y_test, rf_y_pred_proba) if len(set(y_test)) > 1 else 'N/A']
})
st.write(models)