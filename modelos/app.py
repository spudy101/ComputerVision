import cv2
from ultralytics import YOLO
import geocoder
import requests
import base64
import time

# Configuraciones ajustables
TARGET_CLASSES = {
    'cell phone': '0g1egwaIxSuH3e1P2CA5',  # ID de categoría para "Cell Phone"
    'clock': 'dXhs5LKQsiQxIFGBVQVX'        # ID de categoría para "Clock"
}
DETECTION_INTERVAL = 5  # Intervalo de tiempo entre detecciones en segundos

# Cargar el modelo de YOLO
model = YOLO('computerVision/yolov8n.pt')

# Imprimir todas las clases que el modelo reconoce
print("El modelo reconoce las siguientes clases:")
for class_id, class_name in model.names.items():
    print(f"ID: {class_id}, Nombre: {class_name}")

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

        # Filtrar detecciones y solo mostrar las clases objetivo
        for result in results:
            if result.boxes:
                # Crear una lista temporal para almacenar las boxes filtradas
                filtered_boxes = []

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    if class_name in TARGET_CLASSES:
                        filtered_boxes.append(box)

                        category_id = TARGET_CLASSES[class_name]
                        print(f"{class_name.capitalize()} detectado. Procesando...")

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
                            "descripcion": f"{class_name.capitalize()} detectado",
                            "latitud": latitud,
                            "longitud": longitud,
                            "image_data": jpg_as_text,
                            "id_categoria": category_id
                        }

                        # Enviar los datos a la API
                        try:
                            response = requests.post('http://127.0.0.1:5000/api/add_detecciones', json=data)

                            if response.status_code == 200:
                                print(f"Detección enviada a la API con éxito")
                            else:
                                print(f"Error al enviar la detección a la API: {response.status_code}, {response.text}")
                        except requests.exceptions.RequestException as e:
                            print(f"Error al intentar conectar con la API: {e}")

                # Solo dibujar las boxes filtradas en el frame
                for box in filtered_boxes:
                    # Extraer las coordenadas del bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    # Dibujar la caja alrededor del objeto detectado
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Mostrar el frame con las detecciones filtradas
        cv2.imshow('YOLOv8 - Detección', frame)

        # Esperar el intervalo de tiempo configurado antes de continuar
        time.sleep(DETECTION_INTERVAL)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para iniciar la detección y envío a la API
detect_and_send_to_api()
