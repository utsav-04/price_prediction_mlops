import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from sklearn.impute import KNNImputer


logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug('Parameters retrieved from %s', params_path)
        return test_size 
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred while loading the data: %s',e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        imputer = KNNImputer(n_neighbors=5)
        df[['fast_charging', 'num_front_cameras', 'primary_camera_front','extended_upto']] = imputer.fit_transform(df[['fast_charging', 'num_front_cameras', 'primary_camera_front','extended_upto']])
        final_df = df
        logger.debug('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logger.error('Error: An unexpected error occurred during preprocessing: %s',e)
        raise
    except Exception as e:
        logger.error('An unexpected error occurred during preprocessing: %s',e)
        raise

def save_data(train_data: pd.DataFrame,test_data:pd.DataFrame , data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        #train_data.to_csv(os.path.join(data_path, "Y_train.csv"), index=False)
        #Y_test_data.to_csv(os.path.join(data_path, "Y_test.csv"), index=False)
        logger.debug('Data Successfully Saved')
    except Exception as e:
        logger.error('An unexpected error occurred while saving the data: %s',e)
        raise



def main():
    try:
        test_size = load_params(params_path='params.yaml')
        df = load_data(data_url='https://raw.githubusercontent.com/utsav-04/SmartPrix-Data-Scraping-EDA-and-Price-Prediction/refs/heads/main/smartphone_cleaned_v5.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data,test_data, data_path='data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()