# Importación de librerías a utilizar

import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

#Carga de las imágenes

form1=cv2.imread("formulario_01.png")
form1 = cv2.cvtColor(form1, cv2.COLOR_BGR2GRAY)
form2=cv2.imread("formulario_02.png")
form2 = cv2.cvtColor(form2, cv2.COLOR_BGR2GRAY)
form3=cv2.imread("formulario_03.png")
form3 = cv2.cvtColor(form3, cv2.COLOR_BGR2GRAY)
form4=cv2.imread("formulario_04.png")
form4 = cv2.cvtColor(form4, cv2.COLOR_BGR2GRAY)
form5=cv2.imread("formulario_05.png")
form5 = cv2.cvtColor(form5, cv2.COLOR_BGR2GRAY)
form_vacio=cv2.imread("formulario_vacio.png")
form_vacio = cv2.cvtColor(form_vacio, cv2.COLOR_BGR2GRAY)

img=cv2.imread("Imagen_con_detalles_escondidos.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         #hay que tomarla en escala de grises  para que la funcion cv2.equalizeHist pueda realizar la conversión.


#--------------
#PROBLEMA 1
#--------------

def ecualizacion_local(imagen,ventana:int):
    clahe = cv2.createCLAHE (clipLimit=2.0, tileGridSize=(ventana,ventana))
    cl1 = clahe.apply(img) #función de opencv para realizar ecualización local
    ax1=plt.subplot(221)
    plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.subplot(222)
    plt.hist(img.flatten(), 256, [0, 256])
    plt.subplot(223,sharex=ax1,sharey=ax1)
    plt.imshow(cl1,cmap='gray',vmin=0,vmax=255)
    plt.subplot(224)
    plt.hist(cl1.flatten(), 256, [0, 256])
    plt.show()
    return

#Tamaño de la ventana=3
ecualizacion_local(img,3)

#Tamaño de la ventana=6
ecualizacion_local(img,6)

#Tamaño de la ventana=9
ecualizacion_local(img,9)

#Tamaño de la ventana=30
ecualizacion_local(img,30)

#Tamaño de la ventana=80
ecualizacion_local(img,80)

"""
Objetos escondidos:
Recuadro arriba al izquierda= cuadrado
Recuadro arriba a la  derecha= /
Recuadro central= a
Recuadro abajo a la izquierda= lineas horizontales
Recuadro abajo a la derecha= circulo
"""

"""
A medida que se incrementa el tamaño de la ventana, los detalles ocultos dentro de los recuadros negros 
comienzan a apreciarse con mayor nitidez. Sin embargo, cuando el tamaño de la ventana alcanza un valor 
de 80, la imagen empieza a perder definición: los contornos se vuelven más difusos y los colores tienden 
a mezclarse. El fondo deja de mostrarse como un bloque uniforme de gris y comienza a presentar variaciones
en sus tonalidades, mientras que los recuadros negros pierden intensidad y contraste.

"""

#--------------
#PROBLEMA 2 - A
#--------------
      
plt.imshow(form1, cmap='gray'), plt.show(block=False)
plt.imshow(form2, cmap='gray'), plt.show(block=False)
plt.imshow(form3, cmap='gray'), plt.show(block=False)
plt.imshow(form4, cmap='gray'), plt.show(block=False)
plt.imshow(form5, cmap='gray'), plt.show(block=False)



def extraer_celdas_de_imagen(form_img, 
                             bin_thresh=165, 
                             row_thresh_offset=80, 
                             col_thresh_offset=350,
                             min_row_height=3,
                             special_row_start=6,
                             special_row_end=9):
    """
    Detecta y extrae las celdas de una imagen de formulario binarizándola
    y encontrando las líneas horizontales y verticales.

    Args:
        form_img (np.ndarray): La imagen del formulario (ej. 'form1').
        bin_thresh (int): Umbral para binarizar la imagen.
        row_thresh_offset (int): Valor a restar del total de columnas para
                                 detectar líneas horizontales (áreas blancas).
        col_thresh_offset (int): Valor a restar del total de filas para
                                 detectar líneas verticales (áreas blancas).
        min_row_height (int): Altura mínima en píxeles para considerar un
                              rango como un renglón válido.
        special_row_start (int): El número del primer renglón (base 1)
                                 que tiene 3 celdas.
        special_row_end (int): El número del último renglón (base 1)
                               que tiene 3 celdas.

    Returns:
        list: Una lista de listas (celdas_por_renglon), donde cada sublista
              contiene los recortes (np.ndarray) de las celdas para ese renglón.
              
    Raises:
        ValueError: Si no se detectan suficientes divisiones verticales
                    para aplicar la lógica de corte.
    """
    
    # 1. Binarización
    form1_binario = form_img < bin_thresh
    
    # 2. Proyecciones (suma de píxeles)
    form1_binario_cols = np.sum(form1_binario, 0) # Suma vertical (proyección horizontal)
    form1_binario_rows = np.sum(form1_binario, 1) # Suma horizontal (proyección vertical)

    # 3. Thresholding para encontrar las líneas (áreas muy blancas)
    form1_binario_rows_th = form1_binario_rows > (form_img.shape[1] - row_thresh_offset)
    form1_binario_columns_th = form1_binario_cols > (form_img.shape[0] - col_thresh_offset)

    # 4. Encontrar inicios y fines de líneas horizontales
    cambios_filas = np.diff(form1_binario_rows_th.astype(int))
    inicios_y = np.where(cambios_filas == 1)[0] + 1
    fines_y = np.where(cambios_filas == -1)[0] + 1

    lineas_horizontales = list(zip(inicios_y, fines_y))

    # 5. Encontrar inicios y fines de líneas verticales
    cambios_columnas = np.diff(form1_binario_columns_th.astype(int))
    inicios_x = np.where(cambios_columnas == 1)[0] + 1
    fines_x = np.where(cambios_columnas == -1)[0] + 1
    
    lineas_verticales = list(zip(inicios_x, fines_x))

    # 6. Definir rangos de renglones (espacios *entre* líneas horizontales)
    rangos_agrupados = []
    for i in range(len(lineas_horizontales) - 1):
        inicio_rango = lineas_horizontales[i][1] # Fin de la línea actual
        fin_rango = lineas_horizontales[i + 1][0] # Inicio de la sig. línea

        if fin_rango - inicio_rango > min_row_height:
            rangos_agrupados.append((inicio_rango, fin_rango))

    # 7. Cortar la imagen en renglones
    renglones = [form1_binario[y1:y2, :] for y1, y2 in rangos_agrupados]

    # 8. Definir divisiones verticales (espacios *entre* líneas verticales)
    rangos_agrupados_celdas = []
    for i in range(len(lineas_verticales) - 1):
        inicio_rango = lineas_verticales[i][1]
        fin_rango = lineas_verticales[i + 1][0]
        rangos_agrupados_celdas.append((inicio_rango, fin_rango))

    # 9. Validar que existan suficientes divisiones para la lógica
    if len(rangos_agrupados_celdas) < 3:
        raise ValueError(f"⚠️ Se necesitan al menos 3 divisiones verticales detectadas, pero se encontraron {len(rangos_agrupados_celdas)}.")

    # 10. Dividir cada renglón en celdas
    celdas_por_renglon = []
    
    # Puntos de corte basados en el inicio de la 2da y 3ra celda/columna
    corte_1 = rangos_agrupados_celdas[1][0]
    corte_2 = rangos_agrupados_celdas[2][0]

    for renglon_idx, renglon in enumerate(renglones):
        celdas = []
        renglon_num = renglon_idx + 1 # Usar índice base 1 para la lógica

        # División general (2 celdas)
        celda1 = renglon[:, :corte_1]
        celda2_completa = renglon[:, corte_1:] # Esta es la celda que puede subdividirse
        
        # Renglones del 6 al 9 inclusive → 3 divisiones
        if (special_row_start <= renglon_num <= special_row_end):
            # El corte es 'corte_2', pero 'celda2_completa' empieza en 'corte_1'.
            # El corte debe ser relativo al inicio de 'celda2_completa'.
            corte_relativo = corte_2 - corte_1
            
            # Validar que el corte relativo sea válido dentro de la celda
            if 0 < corte_relativo < celda2_completa.shape[1]:
                celda2a = celda2_completa[:, :corte_relativo]
                celda2b = celda2_completa[:, corte_relativo:]
                celdas = [celda1, celda2a, celda2b]
            else:
                # Si el corte no es válido, se añaden las 2 celdas sin subdividir
                print(f"Advertencia: Corte relativo inválido en renglón {renglon_num}. Usando 2 celdas.")
                celdas = [celda1, celda2_completa]
        else:
            # Renglón normal, solo 2 celdas
            celdas = [celda1, celda2_completa]

        celdas_por_renglon.append(celdas)
    
    return celdas_por_renglon


def seleccionar_celdas(formulario) :
    '''
    Esta función recibe un formulario (imagen), llama a la función previamente definida que extrae las celdas
    y crea un diccionario solo con las celdas a utilizar luego para su validación, donde su clave es el nombre
    representativo de lo que contiene la celda, y el valor es la imagen.
    '''
    dicc = {}
    ## llamado a la función que extrae las celdas
    celdas_por_renglon = extraer_celdas_de_imagen(formulario)
    dicc = diccionario_celdas = {
        'nombre_apellido': celdas_por_renglon[1][1],
        'edad':            celdas_por_renglon[2][1],
        'mail':            celdas_por_renglon[3][1],
        'legajo':          celdas_por_renglon[4][1],
        
        # Para las preguntas, guardamos una lista [imagen_si, imagen_no]
        'pregunta_1':      [celdas_por_renglon[6][1], celdas_por_renglon[6][2]],
        'pregunta_2':      [celdas_por_renglon[7][1], celdas_por_renglon[7][2]],
        'pregunta_3':      [celdas_por_renglon[8][1], celdas_por_renglon[8][2]],
        'comentarios':     celdas_por_renglon[9][1]
    }
    return diccionario_celdas


def validar_nombre(imagen_nombre) :
    '''
    Esta función valida la celda "nombre y apellido". Recibe la imagen de dicha celda y detecta los caracteres dentro
    de ella. Se ignoran el fondo y las posibles líneas que hayan quedado del recuadro del formulario, y se realiza
    la validación correspondiente de cantidad de caracteres y palabras, distinguiendolas por espacios.
    '''
    # La imagen que nos llega es binaria (True/False). La pasamos a números (0 y 255).
    img = (imagen_nombre * 255).astype('uint8')

    # La variable más importante que nos da es "stats".
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

    letras_reales = []
    
    # Ignoramos el fondo
    stats_sin_fondo = stats[1:]
    
    # Ignoramos la linea del recuadro
    for mancha in stats_sin_fondo:
        alto = mancha[cv2.CC_STAT_HEIGHT]
        if alto == img.shape[0]:                        # las lineas del recuadro son altas como la imagen
            continue
        
        # Si es más baja que la imagen, es una letra
        letras_reales.append(mancha)
        
    letras_reales = np.array(letras_reales)
    
    # Primer validación (menos de 25 caracteres)
    if len(letras_reales) > 25 or len(letras_reales) < 1 :
        print('Nombre y apellido: MAL')
        return False
    
    # 1. Ordenamos las letras de izquierda a derecha por su coordenada 'x'
    letras_ordenadas = letras_reales[letras_reales[:, cv2.CC_STAT_LEFT].argsort()]
    
    # Asumimos que al menos hay UNA palabra
    cantidad_palabras = 1
    
    # 2. Medimos los huecos entre cada par de letras
    for i in range(len(letras_ordenadas) - 1):
        # El final de la letra actual es su 'x' + su 'ancho'
        fin_letra_actual = letras_ordenadas[i, cv2.CC_STAT_LEFT] + letras_ordenadas[i, cv2.CC_STAT_WIDTH]
        
        # El inicio de la siguiente letra es simplemente su 'x'
        inicio_letra_siguiente = letras_ordenadas[i+1, cv2.CC_STAT_LEFT]
        
        distancia = inicio_letra_siguiente - fin_letra_actual
        
        # 3. Si la distancia es grande (ej. más de 5 píxeles), es un espacio.
        if distancia > 5:
            cantidad_palabras += 1
        
    if cantidad_palabras < 2 :
        print('Nombre y apellido: MAL')
        return False
    
    print('Nombre y Apellido: OK')
    return True
        

def validar_edad (imagen_edad):
    
    # La imagen que nos llega es binaria (True/False). La pasamos a números (0 y 255).
    img = (imagen_edad * 255).astype('uint8')

    # La variable más importante que nos da es "stats".
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

    numeros = []
    
    # Ignoramos el fondo
    stats_sin_fondo = stats[1:]
    
    # Ignoramos la linea del recuadro
    for mancha in stats_sin_fondo:
        alto = mancha[cv2.CC_STAT_HEIGHT]
        if alto == img.shape[0]:                        # las lineas del recuadro son altas como la imagen
            continue
        numeros.append(mancha)
    
    numeros = np.array(numeros)
    
    # Validar cantidad (2 o 3 caracteres)
    if len(numeros) < 2 or len(numeros) > 3 :
        print('Edad: MAL')
        return False
    
    # Validar espacios 
    
    nros_ordenados = numeros[numeros[:, cv2.CC_STAT_LEFT].argsort()]
    
    espacios = ''
    
    # Medimos los huecos entre cada par de letras
    for i in range(len(nros_ordenados) - 1):
        # El final de la letra actual es su 'x' + su 'ancho'
        fin_nro_actual = nros_ordenados[i, cv2.CC_STAT_LEFT] + nros_ordenados[i, cv2.CC_STAT_WIDTH]
        
        # El inicio de la siguiente letra es simplemente su 'x'
        inicio_nro_siguiente = nros_ordenados[i+1, cv2.CC_STAT_LEFT]
        
        distancia = inicio_nro_siguiente - fin_nro_actual
        
        # Si la distancia es grande (ej. más de 5 píxeles), es un espacio.
        if distancia > 5:
            espacios = 'si'
            
    if espacios == 'si':
        print('Edad: MAL')
        return False
    else :
        print('Edad: OK')
        return True    
   

def validar_mail (imagen_mail):
    # La imagen que nos llega es binaria (True/False). La pasamos a números (0 y 255).
    img = (imagen_mail * 255).astype('uint8')

    # La variable más importante que nos da es "stats".
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

    caracteres = []
    
    # Ignoramos el fondo
    stats_sin_fondo = stats[1:]
    
    # Ignoramos la linea del recuadro
    for mancha in stats_sin_fondo:
        alto = mancha[cv2.CC_STAT_HEIGHT]
        if alto == img.shape[0]:                        # las lineas del recuadro son altas como la imagen
            continue
        caracteres.append(mancha)
    
    caracteres = np.array(caracteres)
    
    # Validar cantidad (menos de 25 caracteres)
    if len(caracteres) > 25:
        print('Mail: MAL')
        return False
    
    # Validar espacios 
    
    carac_ordenados = caracteres[caracteres[:, cv2.CC_STAT_LEFT].argsort()]
    
    espacios = 0
    
    # Medimos los huecos entre cada par de letras
    for i in range(len(carac_ordenados) - 1):
        # El final de la letra actual es su 'x' + su 'ancho'
        fin_actual = carac_ordenados[i, cv2.CC_STAT_LEFT] + carac_ordenados[i, cv2.CC_STAT_WIDTH]
        
        # El inicio de la siguiente letra es simplemente su 'x'
        inicio_siguiente = carac_ordenados[i+1, cv2.CC_STAT_LEFT]
        
        distancia = inicio_siguiente - fin_actual
        
        # Si la distancia es grande (ej. más de 5 píxeles), es un espacio.
        if distancia > 5:
            espacios += 1
            
    if espacios > 0:
        print('Mail: MAL')
        return False
    else :
        print('Mail: OK')
        return True


def validar_legajo (imagen_legajo):
    # La imagen que nos llega es binaria (True/False). La pasamos a números (0 y 255).
    img = (imagen_legajo * 255).astype('uint8')

    # La variable más importante que nos da es "stats".
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

    caracteres = []
    
    # Ignoramos el fondo
    stats_sin_fondo = stats[1:]
    
    # Ignoramos la linea del recuadro
    for mancha in stats_sin_fondo:
        alto = mancha[cv2.CC_STAT_HEIGHT]
        if alto == img.shape[0]:                        # las lineas del recuadro son altas como la imagen
            continue
        caracteres.append(mancha)
    
    caracteres = np.array(caracteres)
    
    # Validar cantidad (menos de 25 caracteres)
    if len(caracteres) > 8:
        print('Legajo: MAL')
        return False
    
    # Validar espacios 
    
    carac_ordenados = caracteres[caracteres[:, cv2.CC_STAT_LEFT].argsort()]
    
    espacios = 0
    
    # Medimos los huecos entre cada par de letras
    for i in range(len(carac_ordenados) - 1):
        # El final de la letra actual es su 'x' + su 'ancho'
        fin_actual = carac_ordenados[i, cv2.CC_STAT_LEFT] + carac_ordenados[i, cv2.CC_STAT_WIDTH]
        
        # El inicio de la siguiente letra es simplemente su 'x'
        inicio_siguiente = carac_ordenados[i+1, cv2.CC_STAT_LEFT]
        
        distancia = inicio_siguiente - fin_actual
        
        # Si la distancia es grande (ej. más de 5 píxeles), es un espacio.
        if distancia > 5:
            espacios += 1
            
    if espacios > 0:
        print('Legajo: MAL')
        return False
    else :
        print('Legajo: OK')
    return True


def validar_preguntas(lista, n_pregunta:int):

    cantidad_total = 0
    
    # Iteramos sobre las celdas
    for celda in lista:
    
        # La imagen que nos llega es binaria (True/False). La pasamos a números (0 y 255).
        img = (celda * 255).astype('uint8')

        # La variable más importante que nos da es "stats".
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

        caracteres = []
    
        # Ignoramos el fondo
        stats_sin_fondo = stats[1:]
    
        # Ignoramos la linea del recuadro
        for mancha in stats_sin_fondo:
            alto = mancha[cv2.CC_STAT_HEIGHT]
            if alto == img.shape[0]:                        # las lineas del recuadro son altas como la imagen
                continue
            caracteres.append(mancha)
    
        caracteres = np.array(caracteres)
    
    
        # Validar cantidad (menos de 25 caracteres)
        if len(caracteres) > 1:
            print(f'pregunta {n_pregunta}: MAL')
            return False
        
        if len(caracteres) == 1:
            cantidad_total += 1
        
    if cantidad_total == 0 or cantidad_total > 1:
        print(f'pregunta {n_pregunta}: MAL')
        return False
    else :
        print(f'pregunta {n_pregunta}: OK')
    return True

     
def validar_comentario(imagen_comentario) :
    
    # La imagen que nos llega es binaria (True/False). La pasamos a números (0 y 255).
    img = (imagen_comentario * 255).astype('uint8')

    # La variable más importante que nos da es "stats".
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)

    letras_reales = []
    
    # Ignoramos el fondo
    stats_sin_fondo = stats[1:]
    
    # Ignoramos la linea del recuadro
    for mancha in stats_sin_fondo:
        alto = mancha[cv2.CC_STAT_HEIGHT]
        if alto == img.shape[0]:                        # las lineas del recuadro son altas como la imagen
            continue
        
        # Si es más baja que la imagen, es una letra
        letras_reales.append(mancha)
        
    letras_reales = np.array(letras_reales)
    
    # Primer validación (menos de 25 caracteres)
    if len(letras_reales) > 25 or len(letras_reales) < 1 :
        print('Comentario: MAL')
        return False
    
    print('Comentario: OK')
    return True

#----------------------
#Función Principal
#----------------------

def principal(form, nombre_form:str, form_id:int):
    
    '''
    Esta es la función principal de la validación. Lo único que hace es recibir un formulario, separar las imagenes
    de cada casillero y llamar a cada función de validación según corresponda.
    '''
    
    diccionario = seleccionar_celdas(form)
    lista = []
    
    nya = diccionario['nombre_apellido']
    v_nya = validar_nombre(nya)
    lista.append(v_nya)
    
    edad = diccionario['edad']
    v_edad = validar_edad(edad)
    lista.append(v_edad)
    
    mail = diccionario['mail']
    v_mail = validar_mail(mail)
    lista.append(v_mail)
    
    legajo = diccionario['legajo']
    v_legajo = validar_legajo(legajo)
    lista.append(v_legajo)
    
    pregunta1 = diccionario['pregunta_1']
    v_p1 = validar_preguntas(pregunta1, 1)
    lista.append(v_p1)

    pregunta2 = diccionario['pregunta_2']
    v_p2 = validar_preguntas(pregunta2, 2)
    lista.append(v_p2)

    pregunta3 = diccionario['pregunta_3']
    v_p3 = validar_preguntas(pregunta3, 3)
    lista.append(v_p3)
    
    comentarios = diccionario['comentarios']
    v_coment = validar_comentario(comentarios)
    lista.append(v_coment)
    
    es_valido_todo = all(lista)                     # Comprueba si todos son True
    
    filas_csv = [form_id]                           # Guarda el ID del formulario
    
    for res in lista:
        filas_csv.append('OK' if res else 'MAL')                # Guarda las respuestas según la lista de booleanos
    
    if es_valido_todo:
        print(f'El formulario {nombre_form} está OK')
    else :
        print(f'El formulario {nombre_form} está MAL')
    
    crop_nya_imagen = (nya * 255).astype('uint8')
    
    return crop_nya_imagen, es_valido_todo, filas_csv
    
img1, esvalido1, filas1 = principal(form1, 'Formulario 1', '01')
img2, esvalido2, filas2 = principal(form2, 'Formulario 2', '02')
img3, esvalido3, filas3 = principal(form3, 'Formulario 3', '03')
img4, esvalido4, filas4 = principal(form4, 'Formulario 4', '04')
img5, esvalido5, filas5 = principal(form5, 'Formulario 5', '05')

plt.imshow(img1,cmap='gray'), plt.show()
plt.imshow(img2,cmap='gray'), plt.show()
plt.imshow(img3,cmap='gray'), plt.show()
plt.imshow(img4,cmap='gray'), plt.show()
plt.imshow(img5,cmap='gray'), plt.show()

#--------------
# Problema 2 - B
#--------------

form_A = [(form1, "Formulario 1", "01"), (form2, "Formulario 2", "02"), 
          (form3, "Formulario 3", "03") ]
form_B = [(form4, "Formulario 4", "04"),
        (form5, "Formulario 5", "05")]
formularios = form_A + form_B



def validar_por_tipo():
    '''
    Esta función aplica el algoritmo de validación sobre un tipo de formulario, según sea aclarado en la 
    llamada a la función. 
    '''
    for form, name, id in form_A :
        print(f'Validación del {name}, tipo de formulario A')
        principal(form, name, id)
        
    for form, name, id in form_B :
        print(f'Validación del {name}, tipo de formulario B')
        principal(form, name, id)

    
validar_por_tipo()

#--------------
# PROBLEMA 2 - C    
#--------------


lista_img = []

for form_img, name, id in formularios:
    # Capturamos los 3 valores (asumiendo que tu return es: Imagen, Booleano, Lista)
    crop_nya, es_valido, fila_csv = principal(form_img, name, id)
    
    # Guardamos la imagen (crop) y si es válida (bool)
    lista_img.append( (crop_nya, es_valido) )

if not lista_img:
    print("No hay imágenes para procesar.")
else:
    #1. DEFINIMOS COLORES Y BORDES
    COLOR_OK = (0, 255, 0)   # Verde si es OK
    COLOR_MAL = (0, 0, 255)   # Rojo si es MAL
    grosor_borde = 10
    padding = 20              # Espacio blanco alrededor

    imagenes_listas = []
    ancho_maximo = 0
    alto_total = padding # Empezamos con el padding de arriba

    #2. PRIMER LOOP: Preparamos imágenes y calculamos tamaño total
    for crop_img, es_valido in lista_img:
        
        # A. Convertir a BGR (color) para poder pintar el borde
        if len(crop_img.shape) == 2: # Si es gris
            img_con_borde = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
        else: # Si ya es color
            img_con_borde = crop_img.copy()

        # B. Dibujar el borde
        color_borde = COLOR_OK if es_valido else COLOR_MAL
        h, w = img_con_borde.shape[:2]
        cv2.rectangle(img_con_borde, (0, 0), (w-1, h-1), color_borde, grosor_borde)
        
        # C. Guardar la imagen lista para el segundo loop
        imagenes_listas.append(img_con_borde)
        
        # D. Actualizar tamaños para el lienzo
        alto_total += h + padding # Sumamos el alto de esta img + su padding de abajo
        if w > ancho_maximo:
            ancho_maximo = w # Guardamos el ancho más grande que encontremos

    # 3. CREACIÓN DEL LIENZO
    # Usamos el ancho máximo que encontramos
    total_w = ancho_maximo + (padding * 2) # Ancho max + padding izq y der
    canvas = np.full((alto_total, total_w, 3), 255, dtype=np.uint8)
    
    print(f"Lienzo de: {alto_total} alto x {total_w} ancho")

    #4. SEGUNDO LOOP: Pegamos las imágenes en el lienzo
    y_actual = padding # Empezamos en la 'Y' después del padding de arriba
    
    for img in imagenes_listas:
        h, w = img.shape[:2]
        
        # Pegamos la imagen (que mide 'w' de ancho)
        # El hueco va desde 'padding' hasta 'padding + w'
        canvas[ y_actual : y_actual + h, padding : padding + w ] = img
        
        # Movemos la 'y' para la siguiente imagen
        y_actual += h + padding

    #5. GUARDAMOS LA IMAGEN
    cv2.imwrite("resumen_final.jpg", canvas)
    print("Se guardó la imagen 'resumen_final.jpg'")



#----------------
# PROBLEMA 2 - D
#----------------

lista_para_csv = []

for form_img, name, id in formularios:
    # Capturamos los 3 valores que devuelve la función principal
    crop_nya, es_valido, fila_csv = principal(form_img, name, id)
    lista_para_csv.append(fila_csv)
    

csv_header = [
    "ID", "Nombre y Apellido", "Edad", "Mail", "Legajo", 
    "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"
]


with open('resultados_validacion.csv', 'w', newline = '', encoding = 'utf-8') as f :
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(lista_para_csv)


    

