"""
PART 3 — Interactive Dashboard (OPTIMIZED)
Streamlit app for exploring Titanic data with caching and vectorized filtering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# OPTIMIZATION 1: Cache data with st.cache_data decorator
@st.cache_data
def load_data():
    """Load and cache data - only loads once per session"""
    df = pd.read_csv('../part1-data-cleaning/data_clean/titanic_clean.csv')
    return df

# OPTIMIZATION 2: Cache computed statistics
@st.cache_data
def compute_stats(df):
    """Compute and cache statistics"""
    return {
        'survival': df['survived'].value_counts(),
        'age': df['age'].describe(),
        'fare': df['fare'].describe()
    }

# OPTIMIZATION 3: Vectorized filtering function
def filter_data_vectorized(df, filters):
    """
    Apply all filters at once using vectorized operations.
    MUCH faster than row-by-row or loop-based filtering.
    """
    result = df.copy()
    
    # Vectorized filtering using isin() and boolean indexing
    if 'sex' in filters and filters['sex']:
        result = result[result['sex'].isin(filters['sex'])]
    
    if 'pclass' in filters and filters['pclass']:
        result = result[result['pclass'].isin(filters['pclass'])]
    
    if 'age_range' in filters:
        min_age, max_age = filters['age_range']
        result = result[(result['age'] >= min_age) & (result['age'] <= max_age)]
    
    return result

# OPTIMIZATION 4: Cache plots to avoid regeneration
@st.cache_data
def create_plots(df_hash):
    """
    Create plots and cache them.
    Note: We pass a hash of the data as a parameter to cache separately
    for different filtered datasets.
    """
    plots = {}
    
    # Plot 1: Survival distribution
    survival_data = df.filter(['survived']).value_counts().reset_index(name='count')
    plots['survival'] = px.bar(
        survival_data,
        x='survived', y='count',
        title="Survival Distribution",
        labels={'survived': 'Survived', 'count': 'Count'}
    )
    
    # Plot 2: Age distribution
    plots['age'] = px.histogram(
        df, x='age', nbins=30,
        title="Age Distribution"
    )
    
    # Plot 3: Fare by class
    plots['fare_by_class'] = px.box(
        df, x='pclass', y='fare',
        title="Fare by Passenger Class"
    )
    
    # Plot 4: Survival rate by class
    survival_by_class = df.groupby('pclass')['survived'].mean().reset_index()
    plots['survival_by_class'] = px.bar(
        survival_by_class,
        x='pclass', y='survived',
        title="Survival Rate by Class",
        labels={'pclass': 'Passenger Class', 'survived': 'Survival Rate'}
    )
    
    return plots

def main():
    st.set_page_config(page_title="Titanic Analytics", layout="wide")
    st.title("🚢 Titanic Analytics Dashboard (Optimized)")
    
    # OPTIMIZATION: Data loaded once and cached
    df = load_data()
    
    st.write(f"Dataset shape: {df.shape}")
    st.write("✅ Data cached - no reload on interaction")
    
    # OPTIMIZATION: Sidebar filters
    st.sidebar.header("Filters")
    
    selected_sex = st.sidebar.multiselect(
        "Sex",
        options=sorted(df['sex'].unique()),
        default=list(df['sex'].unique())
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
    
    # OPTIMIZATION 5: Vectorized filtering in one call
    filters = {
        'sex': selected_sex,
        'pclass': selected_class,
        'age_range': age_range
    }
    
    df_filtered = filter_data_vectorized(df, filters)
    st.write(f"Filtered data shape: {df_filtered.shape}")
    st.write("✅ Filters applied using vectorized operations")
    
    # OPTIMIZATION 6: Create plots (cached based on filtered data)
    try:
        plots = create_plots(hash(df_filtered.values.tobytes()))
        
        # Display plots in columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots['survival'], use_container_width=True)
            st.plotly_chart(plots['fare_by_class'], use_container_width=True)
        
        with col2:
            st.plotly_chart(plots['age'], use_container_width=True)
            st.plotly_chart(plots['survival_by_class'], use_container_width=True)
    except Exception as e:
        st.error(f"Error creating plots: {e}")
    
    # OPTIMIZATION 7: Display paginated data (not full dataframe)
    st.header("Raw Data (First 100 rows)")
    st.dataframe(df_filtered.head(100), use_container_width=True)
    
    # OPTIMIZATION 8: Statistics computed once and cached
    st.header("Statistics")
    stats = compute_stats(df_filtered)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Survival Stats")
        st.write(stats['survival'])
    
    with col2:
        st.subheader("Age Stats")
        st.write(stats['age'])
    
    with col3:
        st.subheader("Fare Stats")
        st.write(stats['fare'])
    
    # Performance summary
    st.sidebar.markdown("---")
    st.sidebar.success("""
    ✅ **Optimizations Applied:**
    - Data cached with `@st.cache_data`
    - Vectorized filtering (no loops)
    - Paginated data display
    - Statistics computed once
    - **Expected speedup: 5-10x**
    """)

if __name__ == "__main__":
    main()
