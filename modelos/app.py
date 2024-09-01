import cv2
from ultralytics import YOLO
import geocoder
import requests
import base64
import time

# Configuraciones ajustables
TARGET_CLASS = 'cell phone'  # Clase a detectar (por ejemplo: 'person', 'car', 'dog', etc.)
DETECTION_INTERVAL = 1  # Intervalo de tiempo entre detecciones en segundos

# Cargar el modelo de YOLO
model = YOLO('computerVision/yolov8n.pt')

def get_current_location():
    g = geocoder.ip('me')
    return g.latlng  # Retorna [latitud, longitud]

def detect_and_send_to_api():
    # Inicializa la cámara
    cap = cv2.VideoCapture(0)

    while True:
        print("Leyendo frame de la cámara...")
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # Realizar la detección en el frame actual
        results = model(frame)

        # Imprimir información de las detecciones
        print("Detecciones procesadas:")
        for i, result in enumerate(results):
            print(f"Resultado {i+1}: {result}")
            print(f"Bounding Boxes: {result.boxes}")
            print(f"Clases detectadas: {result.names}")

        # Verificar si alguna de las detecciones es de la clase objetivo
        target_detected = False
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if result.names[class_id] == TARGET_CLASS:
                        target_detected = True
                        print(f"{TARGET_CLASS.capitalize()} detectado. Procesando...")

                        # Codificar el frame actual como imagen JPEG
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if not ret:
                            print("Error al codificar la imagen en formato JPEG.")
                            continue
                        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                        # Obtener ubicación actual
                        location = get_current_location()
                        latitud, longitud = location if location else (None, None)

                        # Datos a enviar a la API
                        data = {
                            "descripcion": f"{TARGET_CLASS.capitalize()} detectado",
                            "latitud": latitud,
                            "longitud": longitud,
                            "image_data": jpg_as_text
                        }

                        # Enviar los datos a la API
                        response = requests.post('http://127.0.0.1:5000/api/add_detecciones', json=data)

                        if response.status_code == 200:
                            print(f"Detección enviada a la API con éxito")
                        else:
                            print(f"Error al enviar la detección a la API: {response.status_code}, {response.text}")

        if not target_detected:
            print(f"No se detectó un(a) {TARGET_CLASS} en este frame.")

        # Mostrar el frame con las detecciones
        cv2.imshow(f'YOLOv8 - Detección de {TARGET_CLASS}', results[0].plot())

        # Esperar el intervalo de tiempo configurado antes de continuar
        time.sleep(DETECTION_INTERVAL)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para iniciar la detección y envío a la API
detect_and_send_to_api()
