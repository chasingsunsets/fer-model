import tensorflow as tf
from tensorflow import keras
import os

# # Load the Keras model
model = tf.keras.models.load_model('model2_epoch25.h5')

# # Convert and save the model as a SavedModel
# tf.saved_model.save(model, 'converted_model/1/')

model_dir = "./model"
model_version = 1
model_export_path = f"{model_dir}/{model_version}"

# tf.saved_model.save(
#     model,
#     export_dir=model_export_path,
# )

print(f"path: {model_export_path}")