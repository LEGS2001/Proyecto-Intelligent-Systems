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


# TODO: ponerle un label a las imagenes de la carpeta screenshots usando labelImg en cmd
# utilizando los labes de los datos entrenados crear un modelo para una CNN
# mostrar los objetos aprendidos en el modelo usadno CV2 (mostrar un video en pantalla con un cuadrado de los objetos)
# objetivo principal del proyecto: mostrar si una persona esta usando un accesorio (gorra, lentes, reloj, etc). usando el modelo entrenado
 
nombre_entrenamiento = "exp2"
model = torch.hub.load('yolov5', 'custom', source='local', path=f'yolov5/runs/train/{nombre_entrenamiento}/weights/best.pt')

print('Modelo cargado')

cam = cv2.VideoCapture(0)
while True:
    ret, vid = cam.read()
    results = model(vid)
    cv2.imshow('YOLO', np.squeeze(results.render()))
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()