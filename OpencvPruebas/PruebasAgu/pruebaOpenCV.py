import cv2

imagen = cv2.imread('formas.png')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gris, 10, 150)

cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##cv2.drawContours(imagen, cnts, -1, (0,255,0), 2)

for c in cnts:
    epsilon = 0.008*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    
    x,y,w,h = cv2.boundingRect(approx)
    if (len(approx)) == 3:
        cv2.putText(imagen, 'Es triangulo Puebla', (x+18,y+150),1,1,(0,0,0),1)

    elif (len(approx)) == 4:
        ratio = float(w)/h
        if(ratio == 1):
            cv2.putText(imagen, 'Es cuadrado Puebla', (x,y+100),1,1,(0,0,0),1)
        else:
            cv2.putText(imagen, 'Es rectangulo Puebla', (x+20,y+100),1,1,(0,0,0),1)
    else:
        ratio = float(w)/h
        if 0.98 <= ratio <= 1.02:
            cv2.putText(imagen, 'Es circulo Puebla', (x+15,y+100),1,1,(0,0,0))
        else:
            cv2.putText(imagen, 'Es ovalo Puebla', (x+20,y+100),1,1,(0,0,0))

    


    print(len(approx))
    cv2.drawContours(imagen, [approx], 0, (0, 0, 0), 2)
    cv2.imshow('imagen', imagen)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

cv2.imshow('Prueba de imagen', imagen)
cv2.waitKey(5000)
##cv2.imshow('Prueba de imagen contornos', canny)
##cv2.waitKey(5000)
cv2.destroyAllWindows()