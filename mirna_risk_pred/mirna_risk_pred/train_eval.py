import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score,confusion_matrix, auc
from tqdm import tqdm

import warnings  
warnings.filterwarnings('ignore')

class MLTools:
    def __init__(self, config):
        self.out_dir = config['out_dir']

    def supervised_pca(self, X_train, y_train, covariates_train, 
                               z_values, T, m):
        
        selected_mirnas = z_values[abs(z_values.iloc[:,1]) > T]['miRNA']
        
        if len(selected_mirnas) < m:
            return None  
        
        
        X_selected = X_train[selected_mirnas]

        pca = PCA(n_components=m)
        X_pca = pca.fit_transform(X_selected)
        X_final = np.hstack([X_pca, covariates_train]) 
        model = LogisticRegression(max_iter=1000)
        model.fit(X_final, y_train)
        return model, pca, selected_mirnas
    

    def find_optimal_parameters(self, X, y, covariates, z_values): 
        T_values = np.arange(0.1, 5.1, 0.1)
        m_values = range(1, 11)
        results = []
        
        # 10-fold CV
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        pbar_t = tqdm(T_values)
        for T in pbar_t:
            for m in m_values: 
                pbar_t.set_description(f"T: {T} - M: {m}") 
                fold_scores = []
                for train_idx, val_idx in skf.split(X, y):
                    # Split data
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    cov_train, cov_val = covariates.iloc[train_idx], covariates.iloc[val_idx]

                  
                    
                    # Apply supervised PCA on training fold
                    result = self.supervised_pca(
                        X_train, y_train, cov_train, z_values, T, m
                    )
                    
                    if result is None:
                        fold_scores.append(0)
                        continue
                    
                    model, pca, selected_mirnas = result
                    
                    # Transform validation data
                    X_val_selected = X_val[selected_mirnas]
                    X_val_pca = pca.transform(X_val_selected)
                    X_val_final = np.hstack([X_val_pca, cov_val])
                    
                    # Predict and calculate AUC
                    y_pred_proba = model.predict_proba(X_val_final)[:, 1]
                    auc = roc_auc_score(y_val, y_pred_proba)
                    fold_scores.append(auc)
                
                # Average AUC across folds
                mean_auc = np.mean(fold_scores) if fold_scores else 0
                
                results.append({
                    'T': T,
                    'm': m,
                    'mean_auc': mean_auc,
                    'n_mirnas': len(z_values[abs(z_values.iloc[:,1]) > T]) if mean_auc > 0 else 0
                })
        
        # Convert to DataFrame and find best
        results_df = pd.DataFrame(results)
        best_idx = results_df['mean_auc'].idxmax()
        best_params = results_df.loc[best_idx]
        
        return best_params, results_df
    
 
    def build_model(self, X_train, y_train, covariates_train, 
                               z_values, best_T, best_m, disease_name):  
        # miRNAs using optimal threshold 
        selected_mirnas = z_values[abs(z_values.iloc[:,1]) > best_T]['miRNA']
        print(f"Selected {len(selected_mirnas)} miRNAs with |z| > {best_T}")
        
        # PCA
        X_selected = X_train[selected_mirnas]
        pca = PCA(n_components=best_m)
        X_pca = pca.fit_transform(X_selected)
        print(f"Applied PCA, kept {best_m} components")
        
        # Combine with clinical variables
        X_final = np.hstack([X_pca, covariates_train])
        
        # model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_final, y_train)
        
        # training results
        y_pred_proba = model.predict_proba(X_final)[:, 1]
        train_auc = roc_auc_score(y_train, y_pred_proba)
        
        # optimal cutoff for classification
        fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # predictions with optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        accuracy = accuracy_score(y_train, y_pred)
        sensitivity = recall_score(y_train, y_pred)
        specificity = recall_score(y_train, y_pred, pos_label=0)
        
        print(f"Model trained, training AUC: {train_auc:.3f}")
        print(f"Optimal threshold: {optimal_threshold:.3f}")
        print(f"-Accuracy: {accuracy:.3f}")
        print(f"-Sensitivity: {sensitivity:.3f}")
        print(f"-Specificity: {specificity:.3f}") 
        
        
        model_package = {
            'model': model,
            'pca': pca,
            'selected_mirnas': list(selected_mirnas),
            'optimal_threshold': optimal_threshold,
            'parameters': {
                'T': best_T,
                'm': best_m,
                'disease': disease_name
            },
            'training_performance': {
                'auc': train_auc,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity
            }
        }
        '''
        
        joblib.dump(model_package, f'{self.data_dir}/final_model_{disease_name}.pkl')
        pd.Series(list(selected_mirnas)).to_csv(f'{self.data_dir}/selected_mirnas_{disease_name}.csv')
        
        # parameters
        with open(f'{self.data_dir}/model_params_{disease_name}.json', 'w') as f:
            json.dump({
                'T': best_T,
                'm': best_m,
                'n_mirnas': len(selected_mirnas),
                'optimal_threshold': optimal_threshold
            }, f, indent=2)
        
        print(f"Final model saved: 'final_model_{disease_name}.pkl'")
        print(f"Selected miRNAs: 'selected_mirnas_{disease_name}.csv'")
        print(f"Parameters: 'model_params_{disease_name}.json'")
        '''
        
        return model_package
     

    def calculate_performance_metrics(self, y_true, y_pred_proba, optimal_threshold): 
        y_pred = (y_pred_proba >= optimal_threshold).astype(int) 
        from sklearn.metrics import accuracy_score, recall_score 
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)  
        specificity = recall_score(y_true, y_pred, pos_label=0)
        
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        pi_cutoff = thresholds[optimal_idx]
        
        return {
            'PI_cutoff': pi_cutoff,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        }
    

    def evaluate_on_validation(self, X_mirna, covariates, y_true, model_package): 
        model = model_package['model']
        pca = model_package['pca']
        selected_mirnas = model_package['selected_mirnas']
        optimal_threshold = model_package['optimal_threshold']
        
        
        X_val_selected = X_mirna[selected_mirnas]
        print(f"Selected  miRNA Counts: {len(selected_mirnas)} miRNAs") 
        X_val_pca = pca.transform(X_val_selected)  
        X_val_final = np.hstack([X_val_pca, covariates]) 
        y_val_pred_proba = model.predict_proba(X_val_final)[:, 1]
        y_val_pred = (y_val_pred_proba >= optimal_threshold).astype(int)
        
        from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
        
        val_auc = roc_auc_score(y_true, y_val_pred_proba)
        val_accuracy = accuracy_score(y_true, y_val_pred)
        val_sensitivity = recall_score(y_true, y_val_pred)
        val_specificity = recall_score(y_true, y_val_pred, pos_label=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_val_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"VALIDATION RESULTS:")
        print(f"- AUC:           {val_auc:.3f}")
        print(f"- Accuracy:      {val_accuracy:.3f}")
        print(f"- Sensitivity:   {val_sensitivity:.3f}")
        print(f"- Specificity:   {val_specificity:.3f}")
        print(f"Confusion Matrix:")
        print(f"- True Negatives:  {tn}")
        print(f"- False Positives: {fp}")
        print(f"- False Negatives: {fn}")
        print(f"- True Positives:  {tp}")
        
        return {
            'auc': val_auc,
            'accuracy': val_accuracy,
            'sensitivity': val_sensitivity,
            'specificity': val_specificity,
            'predictions': y_val_pred_proba,
            'y_true': y_true
        }