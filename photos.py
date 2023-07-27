import cv2
import time
# crea el objeto camara 
cam = cv2.VideoCapture(0) 

# permite recordar en que imagen se quedo para seguir tomando fotos
with open('tools/screenshot_num.txt', 'r') as file:
    screenshot_number = int(file.readline())

while True:
    ret, img = cam.read()
    # muestra la camara capturada por cv2
    cv2.imshow("Camara", img) 
    ch = cv2.waitKey(5)

    # cierra la aplicacion si se presiona la tecla esc
    if ch == 27:
        break

    # toma screenshots mientras se presione la tecla s
    if ch == ord('s'): 
        for i in range(20):
            cv2.imwrite(f'tools/data/screenshots/screenshot{screenshot_number}.png', cam.read()[1])
            time.sleep(0.1)
            screenshot_number += 1
            

# guarda el id de la ultima imagen para poder seguir desde ahi
with open('tools/screenshot_num.txt', 'w') as file:
    file.write(str(screenshot_number))

cv2.destroyAllWindows()