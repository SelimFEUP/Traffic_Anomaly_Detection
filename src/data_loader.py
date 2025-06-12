import pandas as pd
import numpy as np

# Load dataset function
def load_data(filepath):
    data = pd.read_csv(filepath)
    data.drop(columns=['date_hour'], inplace=True)
    # Handle missing values if necessary (e.g., fill forward/backward)
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    return data.values
