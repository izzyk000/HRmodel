import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np

def preprocess_inputs(df, training=True):
    df = df.copy()

    # Drop 'Employee ID' if it exists
    if 'Employee ID' in df.columns:
        df.drop('Employee ID', axis=1, inplace=True)

    if training:
        # Only execute this block if training is True
        # Creating new features for intermediate calculation
        if 'Number of Promotions' in df.columns and 'Years of Service' in df.columns:
            df['Promotion Rate'] = df['Number of Promotions'] / df['Years of Service'].replace(0, np.nan)
            df['Promotion Rate'] = df['Promotion Rate'].fillna(0)

            # Calculate the weighted performance score
            weights = {
                'Promotion Rate': 0.4,
                'Average Annual Reviews Score': 0.6
            }
            df['Performance Score'] = (
                df['Promotion Rate'] * weights['Promotion Rate'] +
                df['Average Annual Reviews Score'] * weights['Average Annual Reviews Score']
            )

            # Define performance categories based on quantiles
            thresholds = df['Performance Score'].quantile([0.2, 0.8])
            df['Performance Category'] = pd.cut(df['Performance Score'], bins=[-np.inf, thresholds.iloc[0], thresholds.iloc[1], np.inf], labels=['Underperform', 'Normal', 'Outperform'])

            # Extract labels before dropping the columns used only for category creation
            y = df['Performance Category'].values

            # Drop intermediate calculation columns and others not needed for modeling
            df.drop(columns=['Years of Service', 'Number of Promotions', 'Promotion Rate', 'Performance Score', 'Average Annual Reviews Score', 'Performance Category'], inplace=True)

    # Define which columns are categorical and which are continuous
    categorical_cols = ['Gender', 'Education Level', 'Role Level', 'Personality Type', 'Origin', 'Previous Company Tier', 'Department', 'Referral']
    continuous_cols = ['Age', 'Years of Working Experience', 'Years of Industry Experience']

    # Create a column transformer
    preprocessor = make_column_transformer(
        (StandardScaler(), continuous_cols),
        (OneHotEncoder(sparse_output=False), categorical_cols),
        remainder='passthrough'
    )

    # Fit and transform the DataFrame if training, else just transform
    if training:
        X_processed = preprocessor.fit_transform(df)
        return X_processed, y, preprocessor
    else:
        X_processed = preprocessor.transform(df)
        return X_processed

