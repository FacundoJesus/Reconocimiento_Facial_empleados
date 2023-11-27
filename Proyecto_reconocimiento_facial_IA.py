import mediapipe as mp # Para el uso de detección de rostros en videocámara
import face_recognition as fr # Para el reconocimiento del rostro
import cv2 # Para la visualización gráfica (rectánculos,etiquetas)
import os # Para el uso de manipular archivos csv
import numpy as np # Para el uso de cálculos matemáticos
from datetime import datetime # Para el uso del tiempo
import locale
from Empleado import Empleado # Importo la clase Empleado

# Variablae para el reconocimiento del rostro (maya facial)
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# ---------------------------------CREAR BASE DE DATOS---------------------------------------
ruta = 'Empleados' # nombre de la carpeta
imagenes_empleados = [] # Lista para las imagenes de los empleados
nombres_empleados = [] # Lista para los nombres de los empleados

# Crear Empleados
empleado_01 = Empleado('Facundo Citera', 40158729, '07-02-1997', 'M', 'Paraná, Entre Ríos', 'Repositor', 'Alto', 3)
empleado_02 = Empleado('Raúl Otamendi', 32526856, '25-03-1990', 'M', 'Paraná, Entre Ríos', 'Recursos humanos','Bajo', 2)
empleado_03 = Empleado('Wenceslao Citera', 42232569, '24-01-2000', 'M', 'Paraná, Entre Ríos', 'Administrador','Medio', 2)

informacion_empleados = [empleado_01,empleado_02,empleado_03] # Lista con la informacion de cada empleado

lista_empleados = os.listdir(ruta)
for nombre in lista_empleados:

    imagen_actual = cv2.imread(f'{ruta}/{nombre}')

    imagenes_empleados.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

#print(imagenes_empleados)
#print(nombres_empleados)
#--------------------------------------------------------------------------------------------

# CODIFICAR IMAGENES
def codificar(imagenes):
    # crear una nueva lista
    lista_codificada = []

    # pasar todas las imagenes de BGR a RGB 
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)

        # codificar
        codificado = fr.face_encodings(imagen)[0]

        # agregar a la lista
        lista_codificada.append(codificado)
    
    # devolver la lista codificada
    return lista_codificada

# Lista de imagenes codificadas...
lista_empleados_codificada = codificar(imagenes_empleados)
#print(len(lista_empleados_codificada))


# Variables datatime
locale.setlocale(locale.LC_TIME, 'es_ES') # Establece la configuración regional a español
ahora = datetime.now() # Obtiene la fecha y hora actual
fecha_hora_espanol = ahora.strftime('%A, %d de %B de %Y - %H:%M:%S') # Formatea la fecha y hora en español


def horario_de_ingreso():
    print(f'Empleado registrado el {fecha_hora_espanol}')
    

def ver_datos_empleado():
    for info_empleado in informacion_empleados:
        if nombre in str(info_empleado):
            print(f'Información del empleado:\n{info_empleado}')  
            

def registrar_ingresos_en_csv(persona):
    archivo = open('registro.csv', 'r+')
    lista_datos = archivo.readlines()

    nombres_registros = []

    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registros.append(ingreso[0])
    
    if persona not in nombres_registros:
        archivo.writelines(f'{persona}, {fecha_hora_espanol}\n')
    


# -------------------------------FLUJO DEL PROGRAMA------------------------------------------------
# TOMAR IMAGEN DE CAMARA WEB
captura = cv2.VideoCapture(0,cv2.CAP_DSHOW) # (id,captura)

contador = 0
carpeta_destino = "./Desconocidos"

"""# Leer el logo
logo = cv2.imread('logo2.png')
filasL, columnasL, canalesL = logo.shape
# Binarizar el logo
logo_gris = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
retm, mascara = cv2.threshold(logo_gris, 190, 255, cv2.THRESH_BINARY)
mascara_inv = cv2.bitwise_not(mascara)
# Operaciones
logo_frente = cv2.bitwise_and(logo, logo, mask = mascara_inv)"""

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,  # Indica la cantidad máxima de rostros detectados. 
    min_detection_confidence=0.5) as face_mesh:

    # Leer frame de la camara continuamente
    while True:

        exito,frame = captura.read()
        if exito == False:
            print('No se pudo realizar la captura de la imagen')
            break

        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        """filasF, columnasF, canalesF = frame.shape
        roi = frame[(filasF-filasL):filasF,(columnasF-columnasL):columnasF]
        # Operaciones
        frame_fondo = cv2.bitwise_and(roi,roi, mask = mascara)
        # Combinar imagenes
        res = cv2.add(frame_fondo, logo_frente)      
        # Añadir la respuesta al frame original
        frame[(filasF-filasL):filasF,(columnasF-columnasL):columnasF] = res"""


        # reconocer cara en captura
        cara_captura = fr.face_locations(frame)
        # codificar cara capturada
        cara_captura_codificada = fr.face_encodings(frame,cara_captura)

        # Código para graficar puntos y conexiones de la maya
        """if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks,
                                          mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                                          )"""


        # buscar coincidencias
        for cara_codif, cara_ubic in zip(cara_captura_codificada, cara_captura):
                                                
            coincidencias = fr.compare_faces(lista_empleados_codificada, cara_codif)
            # puede agregarse la Tolerancia / (lista de empleados con cara codificada y cara codificada para chequear)
            print(f'Lista de coincidencias: {coincidencias}') # Imprime una lista de booleanos: si el rostro es igual a alguna de la BBDD es True

            distancias = fr.face_distance(lista_empleados_codificada, cara_codif)
            # (valor coincidencia: True o False, cuanta distancia hay entre ellas)
            print(f'Lista de distancias de cada imagen de la BBDD: {distancias}') # Imprime una lista de distancias de cada foto
            # DISTANCIA: Compara una foto con otra imagen de la camara para ver que tan distante es esa diferencia.
            # TOLERANCIA: cuanta distancia voy a tolerar para decir que los rotros son los mismos o son de otras personas. por defecto: 0.6
            # mientras menor sea la tolerancia, mas estricto va a ser el programa a la hora de decidir si acepta que es la otra.

            indice_coincidencia = np.argmin(distancias) # obtengo el menor valor de la lista de distancias

            # Mostrar coincidencias si es que las hay...
            if distancias[indice_coincidencia] > 0.6:
                print('*'*120)
                print('No coincide con ninguno de nuestros empleados')

                # Captura el rostro del desconocido
                nombre_archivo = f'captura_{contador}.jpg'
                # Combina la ruta de la carpeta y el nombre del archivo
                ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
                 # Guarda la imagen capturada en la carpeta destino
                cv2.imwrite(ruta_completa, frame)
                print(f'Imagen {nombre_archivo} capturada y guardada en {ruta_completa} el {fecha_hora_espanol}')
                print('*'*120)
                contador += 1
                

                # Cuatro lineas
                y1,x2,y2,x1 = cara_ubic
                # Visualización
                cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2),(0, 0, 255), cv2.FILLED)
                cv2.putText(frame, 'Desconocido',(x1 + 6, y2 - 6),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(frame, face_landmarks,
                                            mp_face_mesh.FACEMESH_TESSELATION,
                                            mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=0), # Puntos
                                            mp_drawing.DrawingSpec(color=(0,0,255,), thickness=1) # Conexiones
                                            )
                
            else:

                # Buscar el nombre del mpleado encontrado
                nombre = nombres_empleados[indice_coincidencia]
                print('*'*82)
                print()
                print(f'Bienvenido al trabajo: "{nombre}"')
                horario_de_ingreso()
                print()
                ver_datos_empleado()
                print('*'*82)


                # Cuatro lineas
                y1,x2,y2,x1 = cara_ubic
                # Visualización
                cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 225, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2),(0, 225, 0), cv2.FILLED,1)
                cv2.putText(frame, nombre,(x1 + 6, y2 - 6),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
                #cv2.rectangle(frame, (510, 350), (630, 470), (255,255,255), 1)

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(frame, face_landmarks,
                                            mp_face_mesh.FACEMESH_TESSELATION,
                                            mp_drawing.DrawingSpec(color=(0,255,0,5), thickness=1, circle_radius=0), # Puntos
                                            mp_drawing.DrawingSpec(color=(0,255,0,5), thickness=1) # Conexiones
                                            )
                
                                           
                # Lo envía y guarda en el archivo .csv
                registrar_ingresos_en_csv(nombre)




        # Muestra el frame con el resultado obtenida
        cv2.imshow('Reconocimiento de empleados', frame)
        
        
        

        # Mantener ventana abierta
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    


# La función release() libera la captura de video o cualquier otro recurso
# relacionado con la fuente de video.
captura.release()  
# Esta función de OpenCV se utiliza para cerrar todas las 
# ventanas creadas por la aplicación
cv2.destroyAllWindows()

