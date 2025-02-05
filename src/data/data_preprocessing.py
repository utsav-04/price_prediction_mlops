import numpy as np
import pandas as pd
import os
import re
import string
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def normalize_text(df):
    try:
        df_encoded = pd.get_dummies(df, columns=['has_5g', 'has_nfc', 'brand_name', 'has_ir_blaster'], drop_first=True)
        logger.debug('completed successfully')
        return df_encoded
        
    except Exception as e:
        logger.error('Error during making dummies %s',e)
        raise
def divide_data_X(df):
    try:
        features = ['model', 'price', 'processor_brand', 'num_cores', 'fast_charging_available',
                    'fast_charging', 'resolution', 'os', 'primary_camera_rear', 'primary_camera_front',
                    'has_5g_True', 'has_nfc_True']
        
        X = df.drop(columns=features)
        return X 
    except Exception as e:
        logger.error('Error occured: %s',e)
        raise 

def divide_data_Y(df):
    try:
        target = 'price'
        y = df[target]
        return y 
    except Exception as e:
        logger.error('Error occured: %s',e)
        raise


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        
        logger.debug('data loaded properly')

        # Transform the data
        test_preprocessed_data = normalize_text(test_data)
        train_preprocessed_data = normalize_text(train_data)
        
        #Y_train_preprocessed_data = normalize_text(train_data)
        #Y_test_preprocessed_data = normalize_text(test_data)

        logger.debug('Train columns after normalize_text: %s', train_preprocessed_data.columns.tolist())
        #logger.debug('Test columns after normalize_text: %s', test_preprocessed_data.columns.tolist())

        X_train_processed_data = divide_data_X(train_preprocessed_data)
        X_test_processed_data = divide_data_X(test_preprocessed_data)
        Y_train_processed_data = divide_data_Y(train_preprocessed_data)
        Y_test_processed_data = divide_data_Y(test_preprocessed_data)


        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        X_train_processed_data.to_csv(os.path.join(data_path, "X_train_processed.csv"), index=False)
        X_test_processed_data.to_csv(os.path.join(data_path, "X_test_processed.csv"), index=False)
        Y_train_processed_data.to_csv(os.path.join(data_path, "Y_train_processed.csv"), index=False)
        Y_test_processed_data.to_csv(os.path.join(data_path, "Y_test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
