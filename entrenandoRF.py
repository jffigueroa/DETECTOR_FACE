import cv2
import os 
import numpy as np

dataPath = 'C:/Users/arzua/IA/face_ID/data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
faceData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imaenes')
    
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        faceData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)
    label = label + 1

# Método de reconocimiento
face_recognizer = cv2.face_LBPHFaceRecognizer.create()

# Entrenando el reconocedor
print("Entrenando... ")
face_recognizer.train(faceData, np.array(labels))

# Almacenando el modelo
face_recognizer.write('modeloLBPHFace.xml')
print("Modelo almacenado...")
