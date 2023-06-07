from .models import models
import yaml

def flatten(array):
    return [item for sublist in array for item in sublist]

def load_model(model_params: dict):
    name, params = model_params.vaues()
    clf = models[name](**params)
    return clf

def read_models(path: str) -> list:
    with open(path, 'rb') as file:
        data = yaml.safe_load(file.read().decode('utf-8'))
    return [load_model(model_data) for model_data in data['ensemble']]

def read_file(path: str) -> str:
     with open(path, 'rb') as file:
         return file.read().decode('utf-8')

