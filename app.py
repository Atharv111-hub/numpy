import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title("🚗 Simple Car Data Analysis")
st.write("A beginner-friendly car dataset explorer")

# Load data from your CSV file
@st.cache_data
def load_car_data():
    """Load car data from cars.csv file"""
    try:
        df = pd.read_csv('cars.csv')
        st.success(f"✅ Successfully loaded {len(df)} cars from cars.csv")
        return df
    except FileNotFoundError:
        st.error("❌ cars.csv file not found. Please make sure the file is in the same folder.")
        st.info("📁 Expected file: cars.csv")
        return None
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

# Load the data
df = load_car_data()

# Stop if data couldn't be loaded
if df is None:
    st.stop()

# Show basic information about the dataset
st.header("📊 Dataset Overview")
st.write(f"Total cars in dataset: **{len(df)}**")
st.write(f"Number of columns: **{len(df.columns)}**")

# Show first few rows
st.subheader("First 5 cars:")
st.dataframe(df.head())

# Show data types
with st.expander("Click to see column details"):
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(col_info)

# Basic statistics
st.header("📈 Basic Statistics")

# Get numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) == 0:
    st.warning("No numeric columns found in your dataset for statistics.")
else:
    # Let user select which columns to analyze
    st.subheader("Select columns for analysis:")
    selected_numeric_cols = st.multiselect(
        "Choose numeric columns:",
        numeric_columns,
        default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns
    )
    
    if selected_numeric_cols:
        col1, col2 = st.columns(len(selected_numeric_cols))
        
        for i, col in enumerate(selected_numeric_cols):
            with [col1, col2][i % 2]:
                st.subheader(f"{col} Statistics")
                st.write(f"Average: {df[col].mean():.2f}")
                st.write(f"Minimum: {df[col].min():.2f}")
                st.write(f"Maximum: {df[col].max():.2f}")
                st.write(f"Missing values: {df[col].isnull().sum()}")
    else:
        st.info("Please select at least one numeric column to see statistics.")

# Simple charts
st.header("📊 Charts")

# Get numeric and categorical columns for charts
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

if len(numeric_columns) > 0:
    # Chart 1: Histogram of numeric column
    st.subheader("Distribution Chart")
    selected_col = st.selectbox("Select column for histogram:", numeric_columns)
    
    fig, ax = plt.subplots()
    ax.hist(df[selected_col].dropna(), bins=20, color='skyblue', alpha=0.7)
    ax.set_xlabel(selected_col)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {selected_col}')
    st.pyplot(fig)

if len(categorical_columns) > 0:
    # Chart 2: Bar chart of categorical column
    st.subheader("Category Count Chart")
    selected_cat = st.selectbox("Select categorical column:", categorical_columns)
    
    cat_counts = df[selected_cat].value_counts().head(10)  # Top 10 categories
    fig, ax = plt.subplots()
    ax.bar(range(len(cat_counts)), cat_counts.values, color='lightgreen')
    ax.set_xticks(range(len(cat_counts)))
    ax.set_xticklabels(cat_counts.index, rotation=45, ha='right')
    ax.set_xlabel(selected_cat)
    ax.set_ylabel('Count')
    ax.set_title(f'Count by {selected_cat}')
    plt.tight_layout()
    st.pyplot(fig)

# Chart 3: Scatter plot if we have at least 2 numeric columns
if len(numeric_columns) >= 2:
    st.subheader("Scatter Plot")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Select X-axis:", numeric_columns, key="x_axis")
    with col2:
        y_col = st.selectbox("Select Y-axis:", numeric_columns, key="y_axis", 
                           index=1 if len(numeric_columns) > 1 else 0)
    
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col], alpha=0.6, color='red')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{y_col} vs {x_col}')
    st.pyplot(fig)

# Interactive filters
st.header("🔍 Filter Cars")

# Create filters based on available columns
filters_applied = False

# Numeric column filters
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_columns) > 0:
    st.subheader("Filter by numeric values:")
    selected_numeric = st.selectbox("Choose a numeric column to filter:", ['None'] + numeric_columns)
    
    if selected_numeric != 'None':
        col_min = float(df[selected_numeric].min())
        col_max = float(df[selected_numeric].max())
        
        if col_min != col_max:  # Only show slider if there's a range
            numeric_range = st.slider(
                f"Select {selected_numeric} range:",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max)
            )
            filters_applied = True
        else:
            st.info(f"All values in {selected_numeric} are the same: {col_min}")

# Categorical column filters
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
if len(categorical_columns) > 0:
    st.subheader("Filter by categories:")
    selected_categorical = st.selectbox("Choose a categorical column to filter:", ['None'] + categorical_columns)
    
    if selected_categorical != 'None':
        unique_values = df[selected_categorical].unique()
        selected_values = st.multiselect(
            f"Select {selected_categorical} values:",
            options=unique_values,
            default=unique_values
        )
        if len(selected_values) < len(unique_values):
            filters_applied = True

# Apply filters
filtered_df = df.copy()

if filters_applied:
    if 'selected_numeric' in locals() and selected_numeric != 'None' and col_min != col_max:
        filtered_df = filtered_df[
            (filtered_df[selected_numeric] >= numeric_range[0]) & 
            (filtered_df[selected_numeric] <= numeric_range[1])
        ]
    
    if 'selected_categorical' in locals() and selected_categorical != 'None':
        filtered_df = filtered_df[filtered_df[selected_categorical].isin(selected_values)]

st.write(f"Showing **{len(filtered_df)}** cars (out of {len(df)} total):")
st.dataframe(filtered_df)

# Fun facts
st.header("🎉 Fun Facts")

numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) > 0:
    # Let user pick a column for fun facts
    fun_fact_col = st.selectbox("Select column for fun facts:", numeric_columns, key="fun_facts")
    
    # Find extreme values
    min_idx = df[fun_fact_col].idxmin()
    max_idx = df[fun_fact_col].idxmax()
    
    # Handle case where we might not have all the columns from original example
    car_info_cols = []
    for col in ['Year', 'Make', 'Model']:
        if col in df.columns:
            car_info_cols.append(col)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Lowest {fun_fact_col}:**")
        if len(car_info_cols) > 0:
            car_desc = " ".join([str(df.loc[min_idx, col]) for col in car_info_cols])
            st.write(car_desc)
        st.write(f"{fun_fact_col}: {df.loc[min_idx, fun_fact_col]}")
    
    with col2:
        st.warning(f"**Highest {fun_fact_col}:**")
        if len(car_info_cols) > 0:
            car_desc = " ".join([str(df.loc[max_idx, col]) for col in car_info_cols])
            st.write(car_desc)
        st.write(f"{fun_fact_col}: {df.loc[max_idx, fun_fact_col]}")
else:
    st.info("No numeric columns available for fun facts.")

