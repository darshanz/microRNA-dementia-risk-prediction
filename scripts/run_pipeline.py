import sys
import os
from pathlib import Path


current_dir = Path.cwd()
project_root = current_dir.parent
src_path = project_root / "src" 
data_dir = f"{project_root.parent}/mirna_data" 
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from data.loaders import DataLoader
from models.supervised_pca import DementiaRiskPredictor
from data.feature_engineering import calculate_z_values
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    loader = DataLoader(data_dir=data_dir)
          
    X, clinical, y = loader.get_data(cohort="discovery", label='ad')
    
    print(f"   Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Positive cases: {y.sum()}, Negative cases: {len(y) - y.sum()}") 


    z_values = calculate_z_values(X, clinical, y)
    print(z_values.shape)
    
    print("\n2. Initializing model...")
    model = DementiaRiskPredictor(
        threshold=3.0,
        n_components=5
    )
    
    print("\n3. Training model...")
    model.fit(X, y, clinical)
    
    print("\n4. Making predictions...")
    predictions = model.predict_proba(X, clinical)
    
    print("\n5. Evaluating model...")
    train_auc = model.evaluate(X, y, clinical)
    print(f"   Training AUC: {train_auc:.3f}")
    
    print("\n6. Saving model...")
    model.save("models/ad_model.pkl")

    print("\n7. Creating visualization...")
    fig = model.plot_feature_importance()
    fig.savefig("results/feature_importance.png", dpi=150, bbox_inches='tight')
    

    print("PIPELINE COMPLETE!")
    print("\nOutputs created:")
    print("  - models/ad_model.pkl (trained model)")
    print("  - results/feature_importance.png")
    print("\nTo use the model for predictions:")
    print("  from supervised_pca import DementiaPredictor")
    print("  model = DementiaPredictor.load('models/ad_model.pkl')")
    print("  predictions = model.predict_proba(new_data, new_clinical)") 
 

if __name__ == "__main__":
    # Create necessary directories
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    main()