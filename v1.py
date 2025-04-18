# Save this as app.py

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import io
from PIL import Image

# ---------------- Setup Section ----------------

# Load the preprocessed data
student_df = pd.read_csv("final_fake_school_data.csv")  # No local path needed on Streamlit Cloud

# Define target and features
target = 'Avg_Student_University_Interest_Score'

top_features_list = [
    'Avg_Student_SPM_Science_Score',
    'Institutional_Engagement_Score',
    'Avg_Student_CoCurricular_Score',
    'Recruitment_Event_Success_Rate',
    'Historical_Enrollment',
    'School_Name',
    'Social_Media_Interactions',
    'Avg_Student_SPM_Math_Score',
    'Preferred_Field_of_Study',
    'Avg_Student_SPM_English_Score'
]

# Select only numeric features
numeric_features = [
    f for f in top_features_list
    if f in student_df.columns and pd.api.types.is_numeric_dtype(student_df[f])
]

features = student_df[numeric_features]

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(features, student_df[target])

# Normalize Features
scaler = MinMaxScaler()
features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

# Calculate dynamic weights based on feature importances
importances = rf.feature_importances_
weights = importances / importances.sum()

for idx, feature in enumerate(features_normalized.columns):
    features_normalized[feature] = features_normalized[feature] * weights[idx]

# Calculate School Visit Priority Score
features_normalized['School_Visit_Priority_Score'] = features_normalized.sum(axis=1)
features_normalized['School_ID'] = student_df['School_ID']
features_normalized['School_Location'] = student_df['School_Location']

# Define Strategy Suggestion Function
def suggest_marketing_strategy(row):
    if row['Avg_Student_SPM_English_Score'] >= 80 and row['Avg_Student_SPM_Science_Score'] >= 75 and row['Institutional_Engagement_Score'] > 0.5:
        return 'Scholarship Invitation + On-site Visit'
    elif row['Social_Media_Interactions'] > 400:
        return 'Social Media Campaigns (Facebook, Instagram)'
    elif row['Recruitment_Event_Success_Rate'] > 30:
        return 'Campus Open Day Invitation'
    elif row['Historical_Enrollment'] > 100:
        return 'High-Volume Event Sponsorship'
    elif row['Institutional_Engagement_Score'] < 0.4:
        return 'Recruitment Event at School'
    else:
        return 'General Digital Marketing'

# Apply strategy suggestion
features_normalized['Recommended_Strategy'] = student_df.apply(suggest_marketing_strategy, axis=1)

# Sort by Priority Score
school_ranking = features_normalized.sort_values(by='School_Visit_Priority_Score', ascending=False)

# SHAP explainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(features, check_additivity=False)

# ---------------- Streamlit UI ----------------

# Load and Display MMU Logo
logo = Image.open("MMU logo.jpg")  # Make sure you upload MMU_logo.jpg into your GitHub repo
col1, col2 = st.columns([1, 5])

with col1:
    st.image(logo, width=100)

with col2:
    st.title("MMU Outreach Prioritization System")

st.write("This system ranks high schools for outreach based on predicted student enrollment potential and recommends suitable marketing strategies.")

st.markdown("---")

# 1. Top 10 Schools
st.header("🏆 Top 10 Schools to Prioritize")
st.dataframe(school_ranking[['School_ID', 'School_Location', 'School_Visit_Priority_Score', 'Recommended_Strategy']].head(10))

st.markdown("---")

# 2. SHAP Feature Importance Summary (Bar Chart)
st.header("📈 Feature Importance (SHAP Summary)")
st.write("The most important factors driving student enrollment potential.")

fig_summary = plt.figure()
shap.summary_plot(shap_values, features, plot_type="bar", show=False)
st.pyplot(fig_summary)

st.markdown("---")

# 3. SHAP Detailed Feature Impact (Beeswarm Plot)
st.header("🧠 Detailed Feature Impact (SHAP Beeswarm)")
fig_swarm = plt.figure()
shap.summary_plot(shap_values, features, show=False)
st.pyplot(fig_swarm)

st.markdown("---")

# 4. Download School Ranking Results
st.header("⬇️ Download School Ranking Results")

# Convert to Excel
output = io.BytesIO()
with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    school_ranking[['School_ID', 'School_Location', 'School_Visit_Priority_Score', 'Recommended_Strategy']].to_excel(writer, index=False, sheet_name='Ranking')
    writer.close()
    processed_data = output.getvalue()

# Create download button
st.download_button(
    label="Download Ranking as Excel",
    data=processed_data,
    file_name='school_ranking.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
