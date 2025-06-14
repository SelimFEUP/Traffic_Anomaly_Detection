import tensorflow as tf
import numpy as np
from src.model import SMAAE
from src.data_loader import load_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, average_precision_score

filepath = 'data/pems.csv'
data = load_data(filepath)
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

model = SMAAE(input_dim=153, latent_dim=32, mem_dim=100)
model.compile(ae_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
model.load_weights('models/model.keras')

def evaluate_model(model, data, contamination=0.01):
    """Comprehensive evaluation of the anomaly detection performance"""
    # Get reconstruction errors
    reconstructions = model(data)
    errors = tf.reduce_mean(tf.square(data - reconstructions), axis=1)
    
    # Get memory attention scores (novel anomaly score)
    z = model.encoder(data)
    z_hat = model.memory(z)
    mem_scores = tf.reduce_mean(tf.square(z - z_hat), axis=1)
    
    # Combined anomaly score
    anomaly_scores = 0.7 * errors + 0.3 * mem_scores
    
    # Threshold based on contamination
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
    
    return anomaly_scores, threshold
