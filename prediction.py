import joblib

def predict(data, model):
    if model == 'Random Forest Classificator':
        model = joblib.load('random_forest_model_regressor.pkl')
    
    pipeline = joblib.load("full_pipeline.pkl")
    data = pipeline.transform(data)
    
    return model.predict(data)