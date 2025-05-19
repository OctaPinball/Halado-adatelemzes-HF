from models import LinearModel, MLPModel
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "../models"
DATA_PATH = "../data"
LOG_PATH = "../logs"
RESULTS_PATH = "../results"

available_models = {
    'linear': lambda input_size: LinearModel(input_size),
    'mlp': lambda input_size: MLPModel(input_size),
    'rfr': lambda input_size: RandomForestRegressor(
        n_estimators=500,
        max_depth=30,
        max_features='sqrt',
        min_samples_split=2,
        random_state=42
    )
}