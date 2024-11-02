import requests

# URL de la API
url = "https://41d8-34-125-79-249.ngrok-free.app/predict/"  # Ajusta el puerto si es necesario

# Ruta de la imagen que deseas enviar
image_path = "../Purpura.jpg"

# Abrir la imagen en modo binario
with open(image_path, "rb") as image_file:
    # Enviar la petición POST con la imagen
    response = requests.post(url, files={"file": image_file})

# Comprobar la respuesta
if response.status_code == 200:
    print("Predicción:", response.json())
else:
    print("Error al hacer la petición:", response.status_code, response.text)
