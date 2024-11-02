import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
model = tf.keras.models.load_model('Models/best_ensemble_model (6).keras')

# Cargar y preprocesar una imagen
image = Image.open('Clorosis.jpg').convert('RGB').resize((256, 256))
img_array = np.array(image).astype(np.float32)
img_array = np.expand_dims(img_array, axis=0)

# Realizar la predicción
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions)

print("Predicciones:", predictions)
print("Clase predicha:", predicted_class)
print("Confianza:", confidence)
