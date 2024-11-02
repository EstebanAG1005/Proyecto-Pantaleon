from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Inicializar la aplicación de FastAPI
app = FastAPI()

# Cargar el modelo de ensamble
model = load_model('../Models/90%Acc.keras')  # Asegúrate de usar la ruta correcta

# Tamaño de imagen esperado por el modelo
IMAGE_SIZE = (224, 224)

# Asumiendo que tienes las clases del modelo
class_names = ['Chinche salivosa', 'Clororis', 'Hoja sana', 'Roya naranja', "Roya purpura"]  # Cambia esto por las clases correctas

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convertir la imagen a RGB si es necesario y cambiar tamaño a IMAGE_SIZE
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        # Preprocesar la imagen
        img_array = np.array(image)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Preprocesamiento específico para MobileNetV2
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch

        # Realizar la predicción
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])  # Índice de la clase con mayor probabilidad
        confidence = np.max(predictions[0])  # Confianza de la predicción

        # Devolver la clase predicha y la confianza
        return {
            "prediction": class_names[predicted_class],
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {str(e)}")

# Para correr la API: uvicorn app:app --reload
