from models import LinearModel, MLPModel

MODEL_PATH = "../models"
DATA_PATH = "../data"
LOG_PATH = "../logs"
RESULTS_PATH = "../results"

available_models = {
    'linear': LinearModel,
    'mlp': MLPModel
}