"""
PART 1 — Data Cleaning
Downloads and cleans Titanic dataset.
"""

import pandas as pd
import os
import time

# Create directories
os.makedirs('data_raw', exist_ok=True)
os.makedirs('data_clean', exist_ok=True)

def download_data():
    """Download Titanic dataset from seaborn"""
    url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
    print("Downloading Titanic dataset...")
    df = pd.read_csv(url)
    df.to_csv('data_raw/titanic.csv', index=False)
    print(f"Downloaded {len(df)} rows")
    return df

def clean_data(df):
    """Clean the dataset - MULTIPLE PERFORMANCE ISSUES HERE"""
    
    # ISSUE 1: Iterating row-by-row to fill missing ages (slow!)
    print("Filling missing ages...")
    for idx, row in df.iterrows():
        if pd.isna(row['age']):
            # Calculate mean age for same pclass
            same_class_mean = df[df['pclass'] == row['pclass']]['age'].mean()
            df.at[idx, 'age'] = same_class_mean
    
    # ISSUE 2: Creating temporary columns repeatedly
    print("Processing gender...")
    for idx, row in df.iterrows():
        if row['sex'] == 'male':
            df.at[idx, 'is_male'] = 1
        else:
            df.at[idx, 'is_male'] = 0
    
    # ISSUE 3: Calling .apply() with expensive function per row
    print("Computing family size...")
    df['family_size'] = df.apply(lambda row: row['sibsp'] + row['parch'] + 1, axis=1)
    
    # ISSUE 4: Filling embarked port by iterating instead of using fillna
    print("Filling embarked ports...")
    for idx, row in df.iterrows():
        if pd.isna(row['embarked']):
            df.at[idx, 'embarked'] = 'S'  # Most common
    
    # ISSUE 5: Duplicate computation - recalculating stats multiple times
    age_mean = df['age'].mean()
    age_std = df['age'].std()
    fare_mean = df['fare'].mean()
    fare_std = df['fare'].std()
    
    print("Normalizing features...")
    for idx, row in df.iterrows():
        if not pd.isna(row['age']):
            df.at[idx, 'age_normalized'] = (row['age'] - age_mean) / age_std
        if not pd.isna(row['fare']):
            df.at[idx, 'fare_normalized'] = (row['fare'] - fare_mean) / fare_std
    
    # ISSUE 6: Creating dummy variables inefficiently
    print("Encoding categories...")
    for col in ['sex', 'embarked', 'pclass']:
        for idx, row in df.iterrows():
            df.at[idx, f'{col}_{row[col]}'] = 1
    
    # ISSUE 7: Dropping columns in a loop
    cols_to_drop = ['name', 'ticket', 'cabin', 'body', 'home.dest']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Remove rows with NaN
    df = df.dropna()
    
    return df

def main():
    start_time = time.time()
    
    # Download and clean
    df = download_data()
    print(f"\nOriginal shape: {df.shape}")
    
    df_clean = clean_data(df)
    print(f"Clean shape: {df_clean.shape}")
    
    # Save
    df_clean.to_csv('data_clean/titanic_clean.csv', index=False)
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    
    return df_clean

if __name__ == "__main__":
    df = main()
