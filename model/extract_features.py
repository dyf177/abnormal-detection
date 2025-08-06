# extract_features.py
from tensorflow.keras.models import Model

def extract_cnn_features(full_model):
    return Model(inputs=full_model.input, outputs=full_model.get_layer('deep_feature').output)
