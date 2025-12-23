
from myworkspaceutils.datasets.micro_rna import MiRNA_GSE120584_Binary
 
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:  
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir) 
        self.mirna = MiRNA_GSE120584_Binary(data_dir)

    def get_data(self, cohort, label):
        meta_data, series_mtrx = self.mirna.load_data(cohort=cohort, label=label)
        y = meta_data[['sample_id','label']].set_index('sample_id')
        X_mirna = series_mtrx.set_index("ID_REF").T
        covariates = meta_data[['sample_id','age', 'sex', 'apoe4']]

        return X_mirna, covariates, y