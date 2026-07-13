"""
PART 3 — Interactive Dashboard
Streamlit app for exploring Titanic data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ISSUE 1: Loading data on every page reload (no caching)
def load_data():
    print("Loading data...")
    df = pd.read_csv('../part1-data-cleaning/data_clean/titanic_clean.csv')
    return df

# ISSUE 2: Computing stats on demand instead of caching
def get_survival_stats(df):
    return df['survived'].value_counts()

def get_age_stats(df):
    return df['age'].describe()

# ISSUE 3: Inefficient filtering without indexing
def filter_data(df, filters):
    result = df.copy()
    
    # ISSUE 4: Filtering in a loop instead of vectorized
    if 'sex' in filters:
        for sex in filters['sex']:
            result = result[result['sex'] == sex]
    
    if 'pclass' in filters:
        for pclass in filters['pclass']:
            result = result[result['pclass'] == pclass]
    
    if 'age_range' in filters:
        min_age, max_age = filters['age_range']
        result = result[(result['age'] >= min_age) & (result['age'] <= max_age)]
    
    return result

# ISSUE 5: Regenerating plots on every interaction
def create_plots(df):
    plots = []
    
    # Plot 1: Survival distribution (regenerated every time)
    plot1 = px.bar(
        df['survived'].value_counts().reset_index(),
        x='survived', y='count',
        title="Survival Distribution"
    )
    plots.append(plot1)
    
    # Plot 2: Age distribution (regenerated every time)
    plot2 = px.histogram(
        df, x='age', nbins=30,
        title="Age Distribution"
    )
    plots.append(plot2)
    
    # Plot 3: Fare by class (regenerated every time)
    plot3 = px.box(
        df, x='pclass', y='fare',
        title="Fare by Passenger Class"
    )
    plots.append(plot3)
    
    # ISSUE 6: Multiple redundant calculations
    # Calculating survival by class multiple times
    survival_by_class = df.groupby('pclass')['survived'].mean()
    plot4 = px.bar(
        survival_by_class.reset_index(),
        x='pclass', y='survived',
        title="Survival Rate by Class"
    )
    plots.append(plot4)
    
    return plots

def main():
    st.set_page_config(page_title="Titanic Analytics", layout="wide")
    st.title("🚢 Titanic Analytics Dashboard")
    
    # ISSUE 7: Reloading data on every interaction
    df = load_data()
    
    st.write(f"Dataset shape: {df.shape}")
    
    # ISSUE 8: Sidebar filters that don't efficiently filter
    st.sidebar.header("Filters")
    
    selected_sex = st.sidebar.multiselect(
        "Sex",
        options=df['sex'].unique(),
        default=df['sex'].unique()
    )
    
    selected_class = st.sidebar.multiselect(
        "Passenger Class",
        options=sorted(df['pclass'].unique()),
        default=sorted(df['pclass'].unique())
    )
    
    age_range = st.sidebar.slider(
        "Age Range",
        float(df['age'].min()),
        float(df['age'].max()),
        (float(df['age'].min()), float(df['age'].max()))
    )
    
    # ISSUE 9: Complex filtering logic that could be vectorized
    filters = {
        'sex': selected_sex,
        'pclass': selected_class,
        'age_range': age_range
    }
    
    df_filtered = filter_data(df, filters)
    
    st.write(f"Filtered data shape: {df_filtered.shape}")
    
    # ISSUE 10: Recreating all plots even if only one filter changed
    plots = create_plots(df_filtered)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plots[0], use_container_width=True)
        st.plotly_chart(plots[2], use_container_width=True)
    
    with col2:
        st.plotly_chart(plots[1], use_container_width=True)
        st.plotly_chart(plots[3], use_container_width=True)
    
    # ISSUE 11: Showing full dataframe without pagination
    st.header("Raw Data")
    st.dataframe(df_filtered, use_container_width=True)
    
    # ISSUE 12: Computing stats again (already computed above)
    st.header("Statistics")
    st.write("Survival Stats:")
    st.write(get_survival_stats(df_filtered))
    st.write("\nAge Stats:")
    st.write(get_age_stats(df_filtered))

if __name__ == "__main__":
    main()
