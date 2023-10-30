import tkinter as tk
import cv2
import os
import imutils

personName = 'user'
dataPath = 'C:/Users/arzua/IA/face_ID/data' 
personPath = dataPath + '/' + personName

if not os.path.exists(personPath):
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)

dataPath = 'C:/Users/arzua/IA/face_ID/data' 
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

# Crear un reconocedor LBPH
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

# Leer el modelo previamente entrenado
face_recognizer.read('modeloLBPHFace.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Declarar caja_texto como variable global
caja_texto = None

# Función que se ejecutará al hacer clic en el botón
def iniciar_sesion():
    while True:
        ret, frame = cap.read()
        if ret == False: 
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
        # LBPHFace
            if result[1] < 70:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if result[1] < 70:
                    cap.release()
                    cv2.destroyAllWindows()
                    os.system("start bala_perdida.mp3")
                    
            else:
                cv2.putText(frame, 'Desconocido', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    resultado.config(text="bienvenido ")
    


def User():
    global caja_texto  # Acceder a la variable caja_texto global
    nombre_usuario = caja_texto.get()  # Obtener el nombre del usuario de la caja de texto
    if not nombre_usuario:
        resultado.config(text="Por favor, ingresa un nombre")
        return

    personPath = dataPath + '/' + nombre_usuario

    if not os.path.exists(personPath):
        print('Carpeta creada: ', personPath)
        os.makedirs(personPath)

    count = 0
    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxFrame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
            count = count + 1
            cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27 or count >= 300:
            break

    cap.release()
    cv2.destroyAllWindows()
    import entrenandoRF
    resultado.config(text="Usuario '{}' Registrado".format(nombre_usuario))
    

    
    
def registrar():
    global caja_texto  # Acceder a la variable caja_texto global
    ventana1 = tk.Tk()
    ventana1.title("Registrarse")
    etiqueta_nombre = tk.Label(ventana1, text="Por favor, ingresa tu nombre:")
    etiqueta_nombre.pack()
    # Caja de texto para ingresar el nombre
    caja_texto = tk.Entry(ventana1, width=30)
    caja_texto.pack()
    boton_saludo = tk.Button(ventana1, text="Registrar", command=User)
    boton_saludo.pack()
    resultado.config(text="Usuario Registrado ")
    
    
    
    
    
    
# Crear una ventana
ventana = tk.Tk()
ventana.title("Iniciar sesión")


# Crear una etiqueta con el texto "Iniciar sesión"
etiqueta = tk.Label(ventana, text="Login")
etiqueta.pack(pady=10)

# Botón para iniciar sesión
boton = tk.Button(ventana, text="Iniciar Sesión", command=iniciar_sesion)
boton.pack()
boton = tk.Button(ventana, text="Registrarse", command=registrar)
boton.pack()

# Etiqueta para mostrar el resultado (puede personalizar esto según sus necesidades)
resultado = tk.Label(ventana, text="")
resultado.pack(pady=10)

# Ejecutar el bucle principal de la ventana
ventana.mainloop()
