import tensorflow as tf
import numpy as np
from src.data_loader import load_data
from src.train import train_model
from src.evaluate import evaluate_model, model

filepath = 'data/pems.csv'
data = load_data(filepath)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Model training
train_model(data=data)

# Evaluate
anomaly_scores, threshold = evaluate_model(model, data, contamination=0.01)
print('Anomaly Scores', anomaly_scores)
