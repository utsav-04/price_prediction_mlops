import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.preprocessing import StandardScaler

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_Scaler(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    """Apply TfIdf to the data."""
    try:
        columns_to_delete = [
        'brand_name_asus', 'brand_name_blackview', 'brand_name_blu', 'brand_name_cat',
        'brand_name_cola', 'brand_name_doogee', 'brand_name_duoqin', 'brand_name_gionee',
        'brand_name_leeco', 'brand_name_leitz', 'brand_name_lenovo', 'brand_name_lyf',
        'brand_name_micromax', 'brand_name_nothing', 'brand_name_nubia', 'brand_name_oukitel',
        'brand_name_sharp', 'brand_name_tcl', 'brand_name_tesla', 'brand_name_vertu'
        ]

        # Drop the columns
        train_data = train_data.drop(columns=columns_to_delete, errors='ignore')
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_data)
        X_test_scaled = scaler.transform(test_data)

        train_data_scaled_df = pd.DataFrame(X_train_scaled, columns=train_data.columns, index=train_data.index)
        test_data_scaled_df = pd.DataFrame(X_test_scaled, columns=test_data.columns, index=test_data.index)

        logger.debug('Scaling of data completed successfully!! ')
        return train_data_scaled_df,test_data_scaled_df
    except Exception as e:
        logger.error('Error during transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        
        X_train_data = load_data('./data/interim/X_train_processed.csv')
        X_test_data = load_data('./data/interim/X_test_processed.csv')
        Y_train_data = load_data('./data/interim/Y_train_processed.csv')
        Y_test_data = load_data('./data/interim/Y_test_processed.csv')

        X_train_df, X_test_df = apply_Scaler(X_train_data, X_test_data)
        #Y_train_df, Y_test_df = apply_Scaler(Y_train_data, Y_test_data)

        data_path = os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)

        X_train_df.to_csv(os.path.join(data_path, "X_train_scalar.csv"),index=False)
        X_test_df.to_csv(os.path.join(data_path, "X_test_scalar.csv"),index=False)
        Y_train_data.to_csv(os.path.join(data_path, "Y_train_scalar.csv"),index=False)
        Y_test_data.to_csv(os.path.join(data_path, "Y_test_scalar.csv"),index=False)
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()