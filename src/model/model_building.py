import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import yaml
import logging

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('Model_Building.log')
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
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, param_dist,cv,n_iter) -> RandomForestRegressor:
    """Train the Logistic Regression model."""
    try:
        rf = RandomForestRegressor(random_state=42)
        # Perform RandomizedSearchCV for hyperparameter tuning
        rf_random = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=cv, random_state=42, n_jobs=-1)
        rf_random.fit(X_train, y_train)
        logger.debug('Model training completed')
        return rf_random.best_estimator_
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        X_train_data = load_data('./data/processed/X_train_Scalar.csv')
        Y_train_data = load_data('./data/processed/Y_train_Scalar.csv')
        params = load_params(params_path='params.yaml')
        logger.debug('params loaded successfully')
        random_search_params = params['model_building']
        hyperparameters = random_search_params['hyperparameters']
        cv = random_search_params['cv']
        n_iter = random_search_params['n_iter']

        clf = train_model(X_train_data, Y_train_data,hyperparameters,cv,n_iter)
        
        save_model(clf, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()