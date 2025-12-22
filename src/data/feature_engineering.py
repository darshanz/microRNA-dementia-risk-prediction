import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import warnings 
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression 


def calculate_z_values(X_mirna, covariates, y):
    z_values = {}
    
    for mirna in tqdm(X_mirna.columns):
        X = pd.concat([X_mirna[[mirna]], covariates.set_index('sample_id')], axis=1)  
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X, y) 

        X_sm = sm.add_constant(X)
        logit_model = sm.Logit(y, X_sm)
        result = logit_model.fit(disp=0)
        
        coef = result.params[1]
        se = result.bse[1]
        z = coef / se
        
        z_values[mirna] = z
    
    return pd.Series(z_values)