import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

st.set_page_config(page_title="Cocktailx HR Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('EA.csv')
    return df

df = load_data()

# SIDEBAR FILTERS
st.sidebar.header("Filters")
customer_segments = df["Customer_Segment"].unique()
segment_choice = st.sidebar.multiselect(
    "Select Customer Segment (Generation):",
    options=customer_segments,
    default=customer_segments
)
st.sidebar.markdown("---")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# FILTER DATA
filtered_df = df[df["Customer_Segment"].isin(segment_choice)]

# MAIN TITLE
st.title("🍹 Cocktailx Customer Insights Dashboard")
st.markdown("""
This dashboard provides HR & stakeholder-focused insights from the Cocktailx dataset, with segment-level and deep-dive analytics. Use the filters on the left to customize your view.
""")

# TABS FOR MACRO & MICRO ANALYSIS
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Macro Trends",
    "2️⃣ Segment Deep-Dive",
    "3️⃣ Feature Relationships",
    "4️⃣ Raw Data & Download"
])

### 1. MACRO TRENDS
with tab1:
    st.header("Macro Overview of Customer Segments")
    st.markdown("""
    **Understand the big picture: How is our customer base distributed by generation and across other major metrics?**
    """)

    # 1. Pie chart: Customer Segment Distribution
    st.markdown("""
    **1. Customer Segment Distribution**  
    A pie chart showing what proportion of customers are from each generation.
    """)
    fig1 = px.pie(filtered_df, names='Customer_Segment', title='Customer Segment Breakdown')
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Bar chart: Count by Segment
    st.markdown("""
    **2. Customer Count by Generation**  
    Visualizes the number of customers in each segment for size comparison.
    """)
    fig2 = px.bar(filtered_df["Customer_Segment"].value_counts().reset_index(),
                  x="index", y="Customer_Segment",
                  labels={"index": "Generation", "Customer_Segment": "Number of Customers"},
                  color="index")
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Gender Distribution
    st.markdown("""
    **3. Gender Proportion by Segment**  
    Examines gender balance within each generation.
    """)
    if "Gender" in filtered_df.columns:
        fig3 = px.histogram(filtered_df, x="Customer_Segment", color="Gender", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

    # 4. Heatmap: Segment vs. Another Categorical (e.g., Occupation)
    if "Occupation" in cat_cols:
        st.markdown("""
        **4. Segment vs. Occupation Heatmap**  
        Reveals which jobs are common in each segment.
        """)
        seg_occ = pd.crosstab(filtered_df["Customer_Segment"], filtered_df["Occupation"])
        fig4, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(seg_occ, annot=True, fmt='d', cmap="Blues", ax=ax)
        st.pyplot(fig4)

### 2. SEGMENT DEEP-DIVE
with tab2:
    st.header("Segment-Level Deep Dives")
    st.markdown("""
    **Analyze each generation’s profile in detail, including spending, preferences, and demographics.**
    """)

    # 5. Select segment for detail
    seg = st.selectbox("Pick a Generation to Deep-Dive:", customer_segments)
    seg_df = filtered_df[filtered_df["Customer_Segment"] == seg]

    # 5. Histogram: Age Distribution
    if "Age" in num_cols:
        st.markdown(f"""
        **5. Age Distribution for {seg}**  
        See the age spread within this segment.
        """)
        fig5 = px.histogram(seg_df, x="Age", nbins=15, title=f'Age Distribution - {seg}')
        st.plotly_chart(fig5, use_container_width=True)

    # 6. Boxplot: Income or Spend by Gender
    for col in ["Income", "Spend", "Annual_Spend"]:
        if col in num_cols and "Gender" in seg_df.columns:
            st.markdown(f"""
            **6. {col} by Gender in {seg}**  
            Are there income/spending differences by gender in this segment?
            """)
            fig6 = px.box(seg_df, x="Gender", y=col, color="Gender", points="all")
            st.plotly_chart(fig6, use_container_width=True)

    # 7. Top 10 Occupations (if available)
    if "Occupation" in cat_cols:
        st.markdown(f"""
        **7. Top Occupations in {seg}**  
        Shows which occupations are most common.
        """)
        top_jobs = seg_df["Occupation"].value_counts().nlargest(10)
        fig7 = px.bar(top_jobs, x=top_jobs.index, y=top_jobs.values,
                      labels={"x": "Occupation", "y": "Count"})
        st.plotly_chart(fig7, use_container_width=True)

    # 8. Average Spend/Income Table
    st.markdown(f"""
    **8. Average Metrics in {seg}**  
    Key average statistics for this segment.
    """)
    if not seg_df[num_cols].empty:
        st.dataframe(seg_df[num_cols].mean().to_frame("Mean Value"))

### 3. FEATURE RELATIONSHIPS
with tab3:
    st.header("Feature Relationships & Patterns")
    st.markdown("""
    **Uncover how different variables relate, including correlations and advanced visualizations.**
    """)

    # 9. Correlation Heatmap (Numerical columns)
    st.markdown("""
    **9. Correlation Matrix (Numerical Features)**  
    See which features move together or have strong relationships.
    """)
    if len(num_cols) > 1:
        fig9, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(filtered_df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig9)

    # 10. Pairplot
    st.markdown("""
    **10. Pairplot (Scatter Matrix)**  
    Visualizes pairwise relationships between top variables.
    """)
    selected = st.multiselect("Choose up to 4 features to pairplot:", num_cols, default=num_cols[:2])
    if len(selected) > 1 and len(selected) <= 4:
        pairplot_fig = sns.pairplot(filtered_df[selected])
        st.pyplot(pairplot_fig.figure)

    # 11. Scatterplot with Filters
    st.markdown("""
    **11. Interactive Scatterplot**  
    Compare any two numeric variables, colored by generation.
    """)
    col_x = st.selectbox("X-Axis:", num_cols, index=0, key="scatter_x")
    col_y = st.selectbox("Y-Axis:", num_cols, index=1, key="scatter_y")
    fig11 = px.scatter(filtered_df, x=col_x, y=col_y, color="Customer_Segment")
    st.plotly_chart(fig11, use_container_width=True)

    # 12. Categorical Count (e.g., Marital Status, if present)
    for cat in ["Marital_Status", "Location", "Channel"]:
        if cat in cat_cols:
            st.markdown(f"""
            **12. Distribution by {cat}**  
            How are generations distributed by {cat}?
            """)
            fig12 = px.histogram(filtered_df, x="Customer_Segment", color=cat, barmode="group")
            st.plotly_chart(fig12, use_container_width=True)

    # 13. KPI Cards
    st.markdown("""
    **13. Key Metrics at a Glance**  
    Instant view of most important figures.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(filtered_df))
    with col2:
        if "Annual_Spend" in num_cols:
            st.metric("Avg Annual Spend", f"{filtered_df['Annual_Spend'].mean():,.2f}")
    with col3:
        if "Income" in num_cols:
            st.metric("Avg Income", f"{filtered_df['Income'].mean():,.2f}")

    # 14. Sunburst or Treemap: Hierarchical View
    if "Location" in cat_cols and "Gender" in cat_cols:
        st.markdown("""
        **14. Hierarchical View: Segment > Location > Gender**  
        Shows distribution across key categories.
        """)
        fig14 = px.sunburst(filtered_df, path=['Customer_Segment', 'Location', 'Gender'])
        st.plotly_chart(fig14, use_container_width=True)

    # 15. Time Trend (if 'Year' or 'Date' column exists)
    for time_col in ["Year", "Join_Year", "Date", "Signup_Date"]:
        if time_col in df.columns:
            st.markdown(f"""
            **15. Customers Over Time**  
            See how customer acquisition or metrics changed by year.
            """)
            fig15 = px.line(filtered_df, x=time_col, y='Customer_Segment', color='Customer_Segment')
            st.plotly_chart(fig15, use_container_width=True)
            break

### 4. RAW DATA & DOWNLOAD
with tab4:
    st.header("Raw Data & Export")
    st.markdown("""
    **Explore the data directly or download for offline analysis.**
    """)
    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data as CSV", csv, "filtered_cocktailx.csv", "text/csv")

# --- ADDITIONAL VISUALS (16-20+) ---
st.markdown("---")
st.subheader("More Insights")

# 16. Violin Plot: Distribution of Spend by Segment
if "Spend" in num_cols:
    st.markdown("""
    **16. Spend Distribution by Generation**  
    See variability and outliers for spend per segment.
    """)
    fig16 = px.violin(filtered_df, y="Spend", x="Customer_Segment", box=True, points="all")
    st.plotly_chart(fig16, use_container_width=True)

# 17. Histogram: Any Numeric Feature
feature = st.selectbox("Pick a numeric feature to histogram:", num_cols, key="hist_feature")
st.markdown("""
**17. Custom Histogram**  
Shows the distribution of your selected feature.
""")
fig17 = px.histogram(filtered_df, x=feature, color="Customer_Segment", nbins=20)
st.plotly_chart(fig17, use_container_width=True)

# 18. Parallel Categories (if multiple categoricals)
if len(cat_cols) >= 3:
    st.markdown("""
    **18. Parallel Categories Plot**  
    Visualizes how different categorical features interact.
    """)
    fig18 = px.parallel_categories(filtered_df, dimensions=cat_cols[:3])
    st.plotly_chart(fig18, use_container_width=True)

# 19. Grouped Bar: Avg Spend/Income by Segment & Gender
for col in ["Spend", "Income"]:
    if col in num_cols and "Gender" in cat_cols:
        st.markdown(f"""
        **19. Avg {col} by Segment & Gender**  
        Compares financial metrics across groups.
        """)
        fig19 = px.bar(filtered_df, x="Customer_Segment", y=col, color="Gender", barmode="group")
        st.plotly_chart(fig19, use_container_width=True)

# 20. Donut Chart: Custom Proportion
if "Loyalty_Level" in cat_cols:
    st.markdown("""
    **20. Loyalty Levels by Generation**  
    Shows customer loyalty breakdown per segment.
    """)
    fig20 = px.pie(filtered_df, names='Loyalty_Level', hole=0.4, color='Customer_Segment')
    st.plotly_chart(fig20, use_container_width=True)

# --- End of Dashboard ---
st.markdown("""
---
*Dashboard built by [Your Name].  
Need help or improvements? Raise an issue on GitHub!*
""")
