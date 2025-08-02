import pandas as pd
import numpy as np

def load_data(filepath):
    filepath = r'C:\Users\ASUS\OneDrive\Desktop\Hospital_Readmission_Prediction\data\diabetes_readmission.csv'
    print("Loading data...")
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def clean_data(df):
    
    print("Cleaning data...")
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop columns with high percentage of missing values and identifiers
    # Weight is missing >97% of values. Payer code and medical specialty have ~50% missing.
    cols_to_drop = ['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr']
    df.drop(columns=cols_to_drop, inplace=True)

    # The 'gender' column has a few 'Unknown/Invalid' values. We will drop these rows.
    df.drop(df[df['gender'] == 'Unknown/Invalid'].index, inplace=True)
    
    # For the target variable 'readmitted', we will create a binary outcome.
    # 1 if readmitted within 30 days, 0 otherwise.
    df['readmitted'] = (df['readmitted'] == '<30').astype(int)

    # Drop other columns that are not useful or have been consolidated
    # examide and citoglipton have only one value 'No' for all records.
    df.drop(columns=['examide', 'citoglipton'], inplace=True)
    
    print(f"Data shape after cleaning: {df.shape}")
    print("Data cleaning complete.")
    return df

if __name__ == '__main__':
    # This block allows you to run this script directly for testing
    file_path = 'diabetes_readmission.csv'
    raw_df = load_data(file_path)
    cleaned_df = clean_data(raw_df.copy())
    print("\nCleaned DataFrame Head:")
    print(cleaned_df.head())
    print("\nInfo of Cleaned DataFrame:")
    cleaned_df.info()
