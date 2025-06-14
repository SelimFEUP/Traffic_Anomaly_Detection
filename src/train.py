import tensorflow as tf
import numpy as np
import pandas as pd
from src.model import SMAAE
from src.data_loader import load_data

def load_model(data):
    model = SMAAE(input_dim=153, latent_dim=32, mem_dim=100)
    model.compile(ae_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='reconstruction_loss', patience=7, mode='min')
    mc = tf.keras.callbacks.ModelCheckpoint('models/model.keras', monitor='reconstruction_loss', verbose=7, save_best_only=True,mode='min')
    
    history = model.fit(data, epochs=100, batch_size=32, verbose=1, callbacks=[mc, early_stopping])
    return history

