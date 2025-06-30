import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv('EA.csv')
    return df

df = load_data()

# ---- SIDEBAR FILTERS ----
st.sidebar.header("Filter Employees")
attrition_options = df['Attrition'].unique()
selected_attrition = st.sidebar.multiselect("Attrition", attrition_options, default=list(attrition_options))

age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

genders = df['Gender'].unique()
selected_gender = st.sidebar.multiselect("Gender", genders, default=list(genders))

departments = df['Department'].unique()
selected_department = st.sidebar.multiselect("Department", departments, default=list(departments))

filtered_df = df[
    (df['Attrition'].isin(selected_attrition)) &
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Gender'].isin(selected_gender)) &
    (df['Department'].isin(selected_department))
]

# ---- MAIN TITLE ----
st.title("HR Dashboard: Employee Attrition Analysis")
st.markdown("""
This dashboard provides a comprehensive overview of employee attrition for HR stakeholders.
Use the interactive filters in the sidebar to explore trends, patterns, and drivers of attrition.
Each section has explanations to guide your interpretation.
""")

# ---- TABS ----
tabs = st.tabs([
    "Overview", "Attrition Breakdown", "Demographics", "Job Satisfaction", "Compensation", 
    "Performance", "Other Factors", "Correlation Matrix", "Prediction (ML)", "Raw Data"
])

# ---- 1. OVERVIEW TAB ----
with tabs[0]:
    st.header("Executive Overview")
    st.markdown("#### Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Employees", len(filtered_df))
    with col2:
        st.metric("Attrition Rate (%)", round(100 * (filtered_df['Attrition'] == 'Yes').sum() / len(filtered_df), 1) if len(filtered_df) else 0)
    with col3:
        st.metric("Avg. Monthly Income", int(filtered_df['MonthlyIncome'].mean()))

    st.markdown("""
    **This section shows a snapshot of your workforce based on filters applied.  
    Attrition rate is a key metric for HR to monitor and reduce employee churn.**
    """)

    # Pie chart: Attrition
    fig1 = px.pie(filtered_df, names='Attrition', title='Attrition Distribution')
    st.plotly_chart(fig1, use_container_width=True)

# ---- 2. ATTRITION BREAKDOWN ----
with tabs[1]:
    st.header("Attrition Breakdown by Categories")
    st.markdown("**Explore how attrition varies across different features.**")

    # Bar: Attrition by Department
    st.markdown("*Attrition by Department helps spot areas with high turnover.*")
    attr_dept = pd.crosstab(filtered_df['Department'], filtered_df['Attrition'])
    st.bar_chart(attr_dept)

    # Bar: Attrition by JobRole
    st.markdown("*Attrition by Job Role highlights critical roles with high exits.*")
    attr_role = pd.crosstab(filtered_df['JobRole'], filtered_df['Attrition'])
    st.bar_chart(attr_role)

    # Bar: Attrition by Gender
    st.markdown("*Is there a gender difference in attrition?*")
    attr_gender = pd.crosstab(filtered_df['Gender'], filtered_df['Attrition'])
    st.bar_chart(attr_gender)

    # Bar: Attrition by EducationField
    st.markdown("*Certain education backgrounds may have higher attrition.*")
    attr_edu = pd.crosstab(filtered_df['EducationField'], filtered_df['Attrition'])
    st.bar_chart(attr_edu)

# ---- 3. DEMOGRAPHICS ----
with tabs[2]:
    st.header("Demographics")
    st.markdown("**Demographic analysis helps in understanding the profile of employees at risk.**")

    # Histogram: Age Distribution
    st.markdown("*Distribution of ages in the current view.*")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['Age'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Boxplot: Age vs. Attrition
    st.markdown("*Age variation by attrition status.*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='Age', data=filtered_df, ax=ax)
    st.pyplot(fig)

    # Countplot: Gender vs. Attrition
    st.markdown("*Gender split among those who left vs. stayed.*")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='Attrition', data=filtered_df, ax=ax)
    st.pyplot(fig)

    # Bar: Marital Status
    st.markdown("*Marital status and its impact on attrition.*")
    marital_attr = pd.crosstab(filtered_df['MaritalStatus'], filtered_df['Attrition'])
    st.bar_chart(marital_attr)

# ---- 4. JOB SATISFACTION ----
with tabs[3]:
    st.header("Job Satisfaction & Environment")
    st.markdown("**Job satisfaction and work-life balance are major drivers of retention.**")

    # Box: JobSatisfaction vs. Attrition
    st.markdown("*Higher job satisfaction typically means lower attrition.*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='JobSatisfaction', data=filtered_df, ax=ax)
    st.pyplot(fig)

    # Bar: WorkLifeBalance
    st.markdown("*Work-life balance rating by attrition status.*")
    wl_balance = pd.crosstab(filtered_df['WorkLifeBalance'], filtered_df['Attrition'])
    st.bar_chart(wl_balance)

    # Bar: EnvironmentSatisfaction
    st.markdown("*Does satisfaction with work environment impact attrition?*")
    env_sat = pd.crosstab(filtered_df['EnvironmentSatisfaction'], filtered_df['Attrition'])
    st.bar_chart(env_sat)

# ---- 5. COMPENSATION ----
with tabs[4]:
    st.header("Compensation & Benefits")
    st.markdown("**Pay and benefits are key motivators. Analyze their effect on attrition.**")

    # Box: MonthlyIncome vs. Attrition
    st.markdown("*Are those leaving paid less?*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='MonthlyIncome', data=filtered_df, ax=ax)
    st.pyplot(fig)

    # Box: PercentSalaryHike vs. Attrition
    st.markdown("*Salary hikes and attrition.*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='PercentSalaryHike', data=filtered_df, ax=ax)
    st.pyplot(fig)

    # Box: TotalWorkingYears vs. Attrition
    st.markdown("*Does tenure affect attrition?*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='TotalWorkingYears', data=filtered_df, ax=ax)
    st.pyplot(fig)

# ---- 6. PERFORMANCE ----
with tabs[5]:
    st.header("Performance & Promotion")
    st.markdown("**Performance management and growth opportunities affect employee retention.**")

    # Bar: PerformanceRating
    st.markdown("*Performance ratings of those who left vs. stayed.*")
    perf_rating = pd.crosstab(filtered_df['PerformanceRating'], filtered_df['Attrition'])
    st.bar_chart(perf_rating)

    # Bar: TrainingTimesLastYear
    st.markdown("*Training and learning opportunities and attrition.*")
    train_attr = pd.crosstab(filtered_df['TrainingTimesLastYear'], filtered_df['Attrition'])
    st.bar_chart(train_attr)

    # Box: YearsSinceLastPromotion vs. Attrition
    st.markdown("*Promotion delays may lead to higher attrition.*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='YearsSinceLastPromotion', data=filtered_df, ax=ax)
    st.pyplot(fig)

    # Box: YearsWithCurrManager vs. Attrition
    st.markdown("*Impact of current manager tenure.*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='YearsWithCurrManager', data=filtered_df, ax=ax)
    st.pyplot(fig)

# ---- 7. OTHER FACTORS ----
with tabs[6]:
    st.header("Other Factors")
    st.markdown("**Explore additional variables impacting attrition.**")

    # Bar: Overtime
    st.markdown("*Do employees doing overtime leave more often?*")
    overtime_attr = pd.crosstab(filtered_df['OverTime'], filtered_df['Attrition'])
    st.bar_chart(overtime_attr)

    # Bar: BusinessTravel
    st.markdown("*Frequent travel may affect retention.*")
    travel_attr = pd.crosstab(filtered_df['BusinessTravel'], filtered_df['Attrition'])
    st.bar_chart(travel_attr)

    # Box: DistanceFromHome vs. Attrition
    st.markdown("*Does distance from home correlate with attrition?*")
    fig, ax = plt.subplots()
    sns.boxplot(x='Attrition', y='DistanceFromHome', data=filtered_df, ax=ax)
    st.pyplot(fig)

# ---- 8. CORRELATION MATRIX ----
with tabs[7]:
    st.header("Correlation Matrix")
    st.markdown("**See which features are most correlated with attrition.**")

    # Encode Attrition for corr
    df_corr = filtered_df.copy()
    df_corr['Attrition_num'] = df_corr['Attrition'].map({'Yes': 1, 'No': 0})

    corr = df_corr.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("*This matrix helps in identifying potential drivers for predictive modeling.*")

# ---- 9. PREDICTION (ML) ----
with tabs[8]:
    st.header("Attrition Prediction (Prototype)")
    st.markdown("""
    **Test a simple machine learning model (Logistic Regression) to predict attrition using selected features.  
    This helps HR teams identify at-risk employees for proactive intervention.**
    """)
    # Simple prototype
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    feature_cols = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'JobSatisfaction']
    X = filtered_df[feature_cols].fillna(0)
    y = filtered_df['Attrition'].map({'Yes': 1, 'No': 0})

    if len(filtered_df) > 50 and y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        st.markdown("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # Confusion matrix
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    else:
        st.warning("Not enough data for model demo. Adjust filters or upload more data.")

# ---- 10. RAW DATA ----
with tabs[9]:
    st.header("Raw Data View")
    st.markdown("**Full employee dataset for custom exploration.**")
    st.dataframe(filtered_df)

# ---- FOOTER ----
st.markdown("""
---
*Dashboard created for HR analytics. For questions, contact your data analytics team.*
""")

