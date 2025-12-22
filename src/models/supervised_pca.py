import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

class DementiaRiskPredictor(BaseEstimator, ClassifierMixin):
    """
    Supervised PCA for dementia prediction. 
    Implements scikit-learn API for compatibility with ML pipelines.
    """
    
    def __init__(self, 
                 threshold: float = 3.0,
                 n_components: int = 5,
                 random_state: int = 42):
        self.threshold = threshold
        self.n_components = n_components
        self.random_state = random_state
        
        # Model components
        self.pca_ = None
        self.classifier_ = None
        self.selected_features_ = None
        self.feature_importances_ = None
        
        logger.info(f"Initialized DementiaRiskPredictor with threshold={threshold}, "
                   f"n_components={n_components}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            clinical_data: Optional[pd.DataFrame] = None) -> 'DementiaRiskPredictor':
        """
        Fit the supervised PCA model.
        
        Parameters
        ----------
        X : miRNA expression data (samples x features)
        y : Target labels
        clinical_data : Clinical covariates (age, sex, APOE)
        
        Returns
        -------
        self : Fitted model
        """
        logger.info(f"Fitting model on {X.shape[0]} samples, {X.shape[1]} features") 
        self.selected_features_ = self._select_features_by_zscore(X, y, clinical_data) # z-scores and select features
        
        #  Apply PCA
        X_selected = X[self.selected_features_]
        self.pca_ = PCA(n_components=self.n_components, 
                       random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_selected)
        
        # clinical data
        if clinical_data is not None:
            X_final = np.hstack([X_pca, clinical_data.values])
        else:
            X_final = X_pca
        
        # Training
        self.classifier_ = LogisticRegression(random_state=self.random_state,
                                             max_iter=1000)
        self.classifier_.fit(X_final, y)
        
        # feature importances
        self._calculate_feature_importances(X_selected)
        
        logger.info(f"Model fitted. Selected {len(self.selected_features_)} features")
        return self
    
    def predict_proba(self, X: pd.DataFrame, 
                     clinical_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Predict probabilities"""
        if self.classifier_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        X_transformed = self._transform_features(X, clinical_data)
        return self.classifier_.predict_proba(X_transformed)
    
    def save(self, path: str):
        """Save model using joblib"""
        import joblib
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load saved model"""
        import joblib
        logger.info(f"Loading model from {path}")
        return joblib.load(path)