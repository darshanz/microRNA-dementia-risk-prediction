import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score,confusion_matrix, auc


def plot_roc_curve(self, val_results, disease='AD'):
        fpr, tpr, thresholds = roc_curve(val_results['y_true'], val_results['predictions'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(4, 3))
        
        # ROC curve
        plt.plot(fpr, tpr, color='black', lw=1,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Diagonal
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        # Optimal point (using your threshold)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        plt.scatter(optimal_fpr, optimal_tpr, color='gray', s=100, 
                    label=f'Optimal threshold\nSens={optimal_tpr:.3f}, Spec={1-optimal_fpr:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'{disease}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return roc_auc


def plot_grid_search_results(self, results_df, disease_name):
    results_df['T'] = round(results_df['T'], 1)
    pivot_table = results_df.pivot(index='m', columns='T', values='mean_auc') 
    plt.figure(figsize=(16, 4))
    sns.heatmap(pivot_table, annot=False, fmt=".2f", cmap="YlOrRd",
                cbar_kws={'label': 'AUC'}) 
    plt.title(f'{disease_name}: AUC for Different (T, m) Combinations', fontsize=14)
    plt.xlabel('Threshold T (z-value)')
    plt.ylabel('PCA Components (m)')
    plt.tight_layout()
    plt.show()