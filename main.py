import torch
import numpy as np
import cv2

# INSTRUCCIONES:

# Notas:
# -El archivo debe estar separado por '-' en caso de tener espacios

# TRAINING
# 1) Correr photos.py, mientras se mantenga la tecla s, se tomarán screenshots para entrenar el modelo

# 2) Abrir labelImg escribiendo labelImg en la consola
#    a) Setearlo en modo de compatibilidad con YOLO
#    b) Elegir los directorios de guardado y de creación de labels en tools > data > screenshots
#    c) Ir creando labels para cada una de las imágenes nombrando cada una de las clases

# 3) Cambiar el archivo dataset.yaml según corresponda
#    a) Ir al archivo en yolov5 > dataset.yaml
#    b) Cambiar nc al número de clases y en name escribir las clases en una lista

# 4) Una vez creado los labels, entrenar el modelo usando yolov5 y pytorch escribiendo el comando en el terminal
#    a) Ir al directirio de yolov5 ---> [cd yolov5]
#    b) python train.py --img 320 --batch 16 --epochs 2000 --data dataset.yaml --hyp hyp.scratch.yaml --weights yolov5s.pt --workers 2
#    c) Esperar a que entrene el modelo


# TODO:
# mostrar los objetos aprendidos en el modelo usadno CV2 (mostrar un video en pantalla con un cuadrado de los objetos)
# objetivo principal del proyecto: mostrar si una persona esta usando un accesorio (gorra, lentes, reloj, etc). usando el modelo entrenado
# hacer que itere sobre todas las caras que encuentre en pantalla y no solo una
# mejorar el modelo: usando mas imagenes, de mejor calidad, y mas epochs
 
nombre_entrenamiento = "exp15"
model = torch.hub.load('yolov5', 'custom', source='local', path=f'yolov5/runs/train/{nombre_entrenamiento}/weights/best.pt')

print('Modelo cargado')

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

other_bboxes = []
cara_bboxes = []

while True:
    status, frame = cam.read()

    results = model(frame)
    df = results.pandas().xyxy[0]
    for i in range(df.shape[0]):
        bbox = df.iloc[i][['xmin','ymin','xmax','ymax']].values.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'], 4)}", (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        if df.iloc[i]['name'] != 'cara':
            #other_bboxes.append([bbox, df.iloc[i]['name']])
            other_bbox = [bbox, df.iloc[i]['name']]

        if df.iloc[i]['name'] == 'cara':
            #cara_bboxes.append(bbox)
            cara_bbox = bbox
        
        if 'cara_bbox' in locals():
            if cara_bbox[0] < other_bbox[0][2] and cara_bbox[2] > other_bbox[0][0] and cara_bbox[1] < other_bbox[0][3] and cara_bbox[3] > other_bbox[0][1]:
                if df.iloc[i]['name'] != 'cara':
                    cv2.putText(frame, f"User is wearing {other_bbox[1]}", (bbox[0], bbox[3] + 15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

        
        #if len(cara_bboxes) > 0: 
        #    for cara_bbox in cara_bboxes:
        #        for other_bbox in other_bboxes:
        #           if cara_bbox[0] < other_bbox[0][2] and cara_bbox[2] > other_bbox[0][0] and cara_bbox[1] < other_bbox[0][3] and cara_bbox[3] > other_bbox[0][1]:
        #               cv2.putText(frame, f"User is wearing {other_bbox[1]}", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
                        


    other_bboxes = []
               
    #cv2.imshow('YOLO', np.squeeze(results.render()))
    cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()