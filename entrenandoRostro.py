import cv2
import os
import imutils
import numpy as np
dataPath = 'C:/Users/arzua/IA/face_ID/data' 
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		image = cv2.imread(personPath+'/'+fileName,0)
	label = label + 1
    
# Metodo de entrenamieno del reconocedor
face_recognizer =cv2.face.LBPHFaceRecognizer_create()

# Entrenamiento del reconocedor de rostro
print("entrenando... ")
face_recognizer.train(facesData, np.array(labels))

# Almacenado del modelo obtenido depues del entrenamiento
face_recognizer.write('mdlLBPHFace.xml')
print("Modelo almacenado con exito")

