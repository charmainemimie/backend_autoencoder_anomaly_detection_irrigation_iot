# model.py
from sklearn.ensemble import IsolationForest
from config import CONTAMINATION

def train_isolation_forest(data):
    features = ['moisture', 'temperature', 'nitrogen', 'phosphorus', 'potassium']
    X = data[features].values
    
    model = IsolationForest(
        contamination=CONTAMINATION,
        random_state=42,
        n_estimators=100,
        n_jobs=-1
    )
    model.fit(X)
    
    return model, features
