import face_recognition # Reconocimimento facial
import cv2 # OpenCV tomara el input de la webcam y va a procesar el input y la va a comparar con la BD
import csv # gestionar archivos csv con datos de la asistencia
import os # acceder a los archivos csv
import numpy as np # algebra lineal
from datetime import datetime # extraer el tiempo actual de la asistencia para el csv

# Codigo
video_capture = cv2.VideoCapture(0) # el argumento va en 0 porque se requiere sacar el input de la camara

# Imagenes
# en caso de tener mas imagenes, se puede hacer con un for esta parte
ning_image = face_recognition.load_image_file("photos/ningning.jpg")
ning_encoding = face_recognition.face_encodings(ning_image)[0]

wendy_image = face_recognition.load_image_file("photos/wendy.jpg")
wendy_encoding = face_recognition.face_encodings(wendy_image)[0]

max_image = face_recognition.load_image_file("photos/max.jpg")
max_encoding = face_recognition.face_encodings(max_image)[0]

andrew_image = face_recognition.load_image_file("photos/andrew.jpg")
andrew_encoding = face_recognition.face_encodings(andrew_image)[0]

known_face_encoding = [
    ning_encoding,
    wendy_encoding,
    max_encoding,
    andrew_encoding
]

known_faces_names = [
    "Ningning",
    "Wendy",
    "Max Changmin",
    "Andrew Garfield"
]

students = known_faces_names.copy()

# Variables para guardar provenientes de OpenCv
face_locations = []
face_encodings = []
face_names = []
s=True 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Creacion del archivo csv
f = open(current_date+'.csv','w',newline='')
lnwriter = csv.writer(f)


while True:
    _,frame = video_capture.read() # extrae los datos de la camara
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25) # decrementa el tamaño del input de la camara en 0.25x0.25
    rgb_small_frame = small_frame[:,:,::-1] # esto es necesario debido a que opencv siempre toma a escala de grises las images, por lo que se requiere pasarlos a rgb

    # Checar la similitud de los rostros
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])

    cv2.imshow("Sistema de Asistencia", frame)
    # Presionar el boton q
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
