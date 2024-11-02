import nest_asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import tensorflow as tf
import numpy as np

# Permitir que FastAPI se ejecute en Jupyter
nest_asyncio.apply()

# Inicializar la aplicación de FastAPI
app = FastAPI()

# Cargar el modelo entrenado de Keras
model = tf.keras.models.load_model('../Models/88%Test.keras')

# Clase para la normalización de las imágenes
class_names = ['Chinche salivosa', 'Clororis', 'Hoja sana', 'Roya naranja', 'Roya purpura']

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convertir la imagen a un formato compatible con el modelo
        image = image.convert("RGB")
        image = image.resize((128, 128))  # Ajustar el tamaño según el modelo
        img_array = np.expand_dims(image, axis=0)  # Añadir dimensión para el batch

        # Realizar la predicción
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_name = class_names[predicted_class]

        return {"prediction": class_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {str(e)}")

# Iniciar el servidor de FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
