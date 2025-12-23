
from mirna_risk_pred.utils import create_logger, read_config_file
from mirna_risk_pred.training_pipeline import TrainingPipeline
 
if __name__ == "__main__":
    config = read_config_file(f'config/config.ini')
    conf = config['run'] 
    logger = create_logger(conf['output_dir'])
   

    pipeline = TrainingPipeline(config=conf)