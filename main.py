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

# carga el modelo entrenado
nombre_entrenamiento = "exp15"
model = torch.hub.load('yolov5', 'custom', source='local', path=f'yolov5/runs/train/{nombre_entrenamiento}/weights/best.pt')
print('Modelo cargado')

# inicializa el objeto de camara
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

# loop principal del programa
while True:
    status, frame = cam.read()

    # predice todos los objetos encontrados en la camara
    results = model(frame)

    # guarda las predicciones en un dataframe de pandas para mejor manejo
    df = results.pandas().xyxy[0]

    # recorre todos las predicciones 
    for i in range(df.shape[0]):

        # guarda el bounding box del objeto en una lista [x1, y1, x2, y2]
        bbox = df.iloc[i][['xmin','ymin','xmax','ymax']].values.astype(int)

        # dibuja un rectangulo sobre cada objeto con su respectivo nombre y confidence
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'], 2)}", (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        # almacena el bounding box y el nombre de los objetos que no sean cara
        if df.iloc[i]['name'] != 'cara':
            #other_bboxes.append([bbox, df.iloc[i]['name']])
            other_bbox = [bbox, df.iloc[i]['name']]

        # almacena el bounding box y el nombre de los objetos que sean cara
        if df.iloc[i]['name'] == 'cara':
            #cara_bboxes.append(bbox)
            cara_bbox = bbox
        
        # detecta si hay colisiones entre los bounding box de cara y accesorio para calcular si el usuario está usando alguno
        if 'cara_bbox' in locals() and 'other_bbox' in locals():
            if cara_bbox[0] < other_bbox[0][2] and cara_bbox[2] > other_bbox[0][0] and cara_bbox[1] < other_bbox[0][3] and cara_bbox[3] > other_bbox[0][1]:
                if df.iloc[i]['name'] != 'cara':
                    cv2.putText(frame, f"Person is wearing {other_bbox[1]}", (other_bbox[0][0], other_bbox[0][3] + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

        #if len(cara_bboxes) > 0: 
        #    for cara_bbox in cara_bboxes:
        #        for other_bbox in other_bboxes:
        #           if cara_bbox[0] < other_bbox[0][2] and cara_bbox[2] > other_bbox[0][0] and cara_bbox[1] < other_bbox[0][3] and cara_bbox[3] > other_bbox[0][1]:
        #               cv2.putText(frame, f"User is wearing {other_bbox[1]}", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
                        
    # muestra la imágen final               
    cv2.imshow('Accessory Detection', frame)

    # al presionar q, cierra el programa
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()