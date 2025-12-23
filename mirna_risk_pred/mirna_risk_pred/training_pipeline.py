import pandas as pd
import numpy as np
from mirna_risk_pred.loaders import DataLoader
from mirna_risk_pred.feature_engineering import calculate_z_values
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mirna_risk_pred.train_eval import MLTools
from mirna_risk_pred.training_pipeline import TrainingPipeline
import logging

class TrainingPipeline:  
    def __init__(self, config=None):
        self.config = config 

    def run(self):
        logging.debug("Loading Data")
        loader = DataLoader(data_dir=self.config['data_dir'])
          
        X, clinical, y = loader.get_data(cohort="discovery", label='ad')
        z_values = calculate_z_values(X, clinical, y) 
        mltools = MLTools(self.config)
        
        
        X = X.select_dtypes(include=[np.number])
        clinical = clinical.select_dtypes(include=[np.number])
        y = y.select_dtypes(include=[np.number])
        best_params, results_df = mltools.find_optimal_parameters(X, y, clinical, z_values)

        model = mltools.build_model(
            X_train=X, 
            y_train=y, 
            covariates_train=clinical, 
            z_values=z_values['z'].values,
            best_T=best_params['T'],
            best_m=int(best_params['m']),
            disease_name='AD')

        X_val, clinical_val, y_val = loader.get_data(cohort="validation", label='ad')
        results  = mltools.evaluate_on_validation(X_val, clinical_val, y_val, model)
        print(results)