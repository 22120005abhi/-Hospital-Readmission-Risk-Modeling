import pandas as pd
import numpy as np

def map_diagnosis_codes(df):
    """
    Groups diagnosis codes (diag_1, diag_2, diag_3) into broader categories.
    This simplifies the feature space and captures more general health conditions.
    """
    print("Engineering diagnosis features...")
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        # Ensure the column is of a string type before using .str accessor
        df[col] = df[col].astype(str).str.replace('E', '').str.replace('V', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Grouping based on ICD-9 main chapters
    # FIX: Dictionary keys changed from lists to tuples
    diag_map = {
        tuple(list(range(1, 140)) + [780, 781, 782, 784, 790, 791, 793, 794, 795, 796, 798, 799]): 'Infectious',
        tuple(range(140, 240)): 'Neoplasms',
        (250,): 'Diabetes', # Use a tuple for a single value as well
        tuple(list(range(240, 250)) + list(range(251, 280))): 'Endocrine/Metabolic',
        tuple(range(280, 290)): 'Blood',
        tuple(range(290, 320)): 'Mental',
        tuple(range(320, 390)): 'Nervous',
        tuple(list(range(390, 460)) + [785]): 'Circulatory',
        tuple(list(range(460, 520)) + [786]): 'Respiratory',
        tuple(list(range(520, 580)) + [787]): 'Digestive',
        tuple(list(range(580, 630)) + [788]): 'Genitourinary',
        tuple(range(630, 680)): 'Pregnancy',
        tuple(range(680, 710)): 'Skin',
        tuple(range(710, 740)): 'Musculoskeletal',
        tuple(range(740, 760)): 'Congenital',
        tuple(range(800, 1000)): 'Injury',
        tuple([783, 789, 792, 797] + list(range(760, 780))): 'Other'
    }


    for col in diag_cols:
        new_col_name = f'{col}_category'
        df[new_col_name] = 'Other' # Default category
        # FIX: Iterate through tuple keys
        for codes_tuple, category in diag_map.items():
            df.loc[df[col].isin(codes_tuple), new_col_name] = category
            
    # We can now drop the original diagnosis columns
    df.drop(columns=diag_cols, inplace=True)
    return df


def create_features(df):
    """
    Creates new features to improve model performance.

    Args:
        df (pandas.DataFrame): The input dataframe (should be cleaned).

    Returns:
        pandas.DataFrame: Dataframe with new features.
    """
    print("Starting feature engineering...")
    
    # Map diagnosis codes
    df = map_diagnosis_codes(df)

    # Convert age brackets to numerical
    print("Engineering age feature...")
    age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, 
               '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
    df['age'] = df['age'].map(age_map)

    # Feature for number of medication changes
    print("Engineering medication features...")
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
                'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    df['num_med_changes'] = 0
    for col in med_cols:
        # This check handles any non-string values gracefully
        df['num_med_changes'] += df[col].isin(['Up', 'Down'])

    # Service utilization feature
    print("Engineering service utilization feature...")
    df['service_utilization'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

    # One-hot encode categorical features
    print("One-hot encoding categorical features...")
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Use pd.get_dummies for one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Feature engineering complete.")
    return df_encoded

if __name__ == '__main__':
    # This block allows you to run this script directly for testing
    from data_preparation import load_data, clean_data
    
    file_path = 'diabetes_readmission.csv'
    raw_df = load_data(file_path)
    cleaned_df = clean_data(raw_df.copy())
    featured_df = create_features(cleaned_df.copy())
    
    print("\nFeatured DataFrame Head:")
    print(featured_df.head())
    print(f"\nShape of final featured DataFrame: {featured_df.shape}")
    print("\nColumns in featured DataFrame:")
    print(featured_df.columns.tolist())
