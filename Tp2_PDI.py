import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#----------------------
#Problema 1 - Monedas y Dados
#----------------------

# --- Lectura de imagen --------------------------------------------------
img = cv2.imread('monedas.jpg',cv2.IMREAD_GRAYSCALE)

img.shape #Dimensiones de la imagen.
type(img) #Tipo de dato de c/pixel

img.min()
img.max()
plt.figure(), plt.imshow(img, cmap="gray"), plt.show(block=False)

# --- Detección automática de monedas y dados -----------------------
img_blur = cv2.GaussianBlur(img, (3,3), 2)
plt.figure()
ax1 = plt.subplot(121); plt.imshow(img, cmap="gray"), plt.title("Imagen")
plt.subplot(122, sharex=ax1, sharey=ax1), plt.imshow(img_blur, cmap="gray"), plt.title("Imagen + blur")
plt.show(block=False)

img_canny = cv2.Canny(img_blur, 50, 150, apertureSize=3, L2gradient=True)

plt.figure(), plt.imshow(img_canny, cmap="gray"), plt.show(block=False)

# --- Dilatacion -----------------------
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
img_canny_d = cv2.dilate(img_canny, kernel, iterations=1)

plt.figure(), plt.imshow(img_canny_d, cmap="gray"), plt.show(block=False)

def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                             
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   
        if (marker == expanded_intersection).all():                         
            break                                                           
        marker = expanded_intersection        
    return expanded_intersection

# Utilizando cv2.floodFill()
# SI rellena los huecos que tocan los bordes
def imfillhole_v2(img):
    img_flood_fill = img.copy().astype("uint8")             # Genero la imagen de salida
    h, w = img.shape[:2]                                    # Genero una máscara necesaria para cv2.floodFill()
    mask = np.zeros((h+2, w+2), np.uint8)                   # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#floodfill
    cv2.floodFill(img_flood_fill, mask, (0,0), 255)         # Relleno o inundo la imagen.
    img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)    # Tomo el complemento de la imagen inundada --> Obtenog SOLO los huecos rellenos.
    img_fh = img | img_flood_fill_inv                       # La salida es un OR entre la imagen original y los huecos rellenos.
    return img_fh 

img_fh2 = imfillhole_v2(img_canny_d)
plt.figure(), plt.imshow(img_fh2 , cmap="gray"), plt.show(block=False)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
Aop = cv2.morphologyEx(img_fh2, cv2.MORPH_OPEN, B)
plt.figure(), plt.imshow(Aop , cmap="gray"), plt.show(block=False)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Aop, 4, cv2.CV_32S)

# Obtener contornos para cada componente
Aop_uint8 = Aop.astype("uint8")
contours, hierarchy = cv2.findContours(Aop_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

monedas = []
dados = []

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        continue

    circularidad = 4 * np.pi * area / (perimeter * perimeter)

    # Clasificación según circularidad
    if circularidad > 0.85:  
        monedas.append(i)
    else:
        dados.append(i)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularidad = 4*np.pi*area/(perimeter*perimeter)

    color = (0,255,0) if circularidad>0.85 else (0,0,255)
    cv2.drawContours(img_color, [cnt], -1, color, 6)

plt.figure(); plt.imshow(img_color[:,:,::-1]); plt.show()

#Verde->Moneda
#Rojo->Dado

# --- Clasificación de monedas -----------------------
areas_monedas = []

for i in monedas:   
    area = cv2.contourArea(contours[i])
    areas_monedas.append(area)

print("Áreas detectadas de monedas:", areas_monedas) 

clasificacion_monedas = []
conteo = {
    "chica": 0,
    "mediana": 0,
    "grande": 0
}

for idx in monedas:
    cnt = contours[idx]
    area = cv2.contourArea(cnt)

    # Clasificación según umbrales
    if area < 80000:
        tipo = "chica"
    elif 100000 <= area < 110000:
        tipo = "mediana"
    elif area >= 110000:
        tipo = "grande"
    else:
        tipo = "no_clasificada"

    clasificacion_monedas.append((idx, area, tipo))

    if tipo in conteo:
        conteo[tipo] += 1

# Resultados
print("Clasificación de monedas:")
for i, area, tipo in clasificacion_monedas:
    print(f"Moneda {i} | Área={area:.0f} | Tipo={tipo}")
    
print(conteo)


# --- Número de puntos de cara superior del dado ---
numeros_dados = {}

for idx in dados:
    # Bounding box del dado
    cnt = contours[idx]
    x, y, w, h = cv2.boundingRect(cnt)
    roi = img[y:y+h, x:x+w].copy()      # ROI original

    # --- 1) Suavizado ---
    blur = cv2.GaussianBlur(roi, (7,7), 0)

    # --- 2) Threshold adaptativo ---
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # --- 3) Quitar ruido ---
    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- 4) Contornos ---
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 5) Filtrar por área ---
    contornos_filtrados = [
        c for c in cnts
        if 500 < cv2.contourArea(c) < 4000
    ]

    print("Áreas filtradas:", [cv2.contourArea(c) for c in contornos_filtrados])

    # ROI ORIGINAL
    plt.figure(figsize=(4,4))
    plt.imshow(roi, cmap="gray")
    plt.title(f"Dado {idx} - ROI original")
    plt.axis("off")
    plt.show(block=False)

    # --- 6) Contar puntos ---
    numero_detectado = len(contornos_filtrados)
    print(f"Dado {idx} → Número detectado: {numero_detectado}")


    numeros_dados[idx] = numero_detectado



print("RESULTADO FINAL DADOS")
for dado, valor in numeros_dados.items():
    print(f"Dado {dado}: {valor} puntos")

#----------------------
#Problema 2 - Patentes
#----------------------
patentes = []

for i in range(1,13):
    archivo = f'img{i}.png'
    patente = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
    plt.figure(), plt.imshow(patente, cmap="gray"), plt.show(block=False)
    patentes.append(patente)
    
kernel = np.ones((2,2), np.uint8)
imagenes_bin = []

for i, patente in enumerate(patentes):
    img_bin = cv2.adaptiveThreshold(patente, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 9, 3)
    if i == 9:
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    #    plt.figure(), plt.imshow(img_bin, cmap="gray"), plt.show(block=False)
    if i == 5:
        img_bin = cv2.erode(img_bin, kernel, iterations=1)
    #    plt.figure(), plt.imshow(img_bin, cmap="gray"), plt.show(block=False)
    imagenes_bin.append(img_bin)

#--------------------------------------------------------------------------------------------
## 3. CONNECTED COMPONENTS Y FILTRADO POR ASPECTO, ÁREA Y CERCANÍA

DISTANCIA_MAXIMA = 80.0
imagenes_filtradas_final = [] # Guarda la imagen binaria con solo los caracteres finales
bouding_boxes_validos = []    # Guarda la lista de BBoxes de los caracteres válidos por imagen

for i, img in enumerate(imagenes_bin):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    
    # 3.1. Primer Filtro: Área y Ratio
    indices_validos_por_aspecto = []
    
    for j in range(1, num_labels):
        ancho = stats[j, cv2.CC_STAT_WIDTH]
        alto = stats[j, cv2.CC_STAT_HEIGHT]
        area = stats[j, cv2.CC_STAT_AREA]
        
        if ancho == 0 or alto == 0:
            continue
            
        relacion = float(alto) / ancho
        
        es_letra_valida = (area >= 20 and area <= 110) and (relacion >= 1.5 and relacion <= 3.0)
        
        if es_letra_valida:
            indices_validos_por_aspecto.append(j)

    # 3.2. Segundo Filtro: Cercanía de Centroides
    imagen_filtrada_cercania = np.zeros_like(img)
    bouding_boxes_caracteres = [] # Lista temporal para guardar los BBoxes de los caracteres válidos
    
    for idx_i in indices_validos_por_aspecto:
        centroide_actual = centroids[idx_i]
        tiene_vecino_cerca = False
        
        for idx_j in indices_validos_por_aspecto:
            if idx_i == idx_j: continue
                
            centroide_vecino = centroids[idx_j]
            distancia = math.sqrt((centroide_actual[0] - centroide_vecino[0])**2 + 
                                  (centroide_actual[1] - centroide_vecino[1])**2)
            
            if distancia < DISTANCIA_MAXIMA:
                tiene_vecino_cerca = True
                break 
        
        if tiene_vecino_cerca:
            imagen_filtrada_cercania[labels == idx_i] = 255
            bbox_stats = stats[idx_i, cv2.CC_STAT_LEFT:cv2.CC_STAT_AREA] 
            bouding_boxes_caracteres.append(bbox_stats)
            
    # Guardamos los resultados para las fases siguientes
    imagenes_filtradas_final.append(imagen_filtrada_cercania)
    bouding_boxes_validos.append(bouding_boxes_caracteres)

    # 3.3. Visualización Intermedia
    plt.figure()
    plt.imshow(imagen_filtrada_cercania, cmap="gray")
    plt.title(f"Img {i+1}: Fase 3 - Filtro por Aspecto y Cercanía")
    plt.show(block=False)


    
#--------------------------------------------------------------------------------------------

## 4, 5, 6, 7. BOUNDING BOX Y RECORTE

# Guardamos las patentes completas recortadas
recortes_patentes_completas = [] 

# Iteramos sobre la lista de imágenes binarias filtradas
for i, img_filtrada in enumerate(imagenes_filtradas_final):
    archivo = f'img{i+1}.png' 
    img_color = cv2.imread(archivo, cv2.IMREAD_COLOR) 
    
    # 4. Bounding Box a la Patente Completa
    puntos_blancos = cv2.findNonZero(img_filtrada)
    
    if puntos_blancos is not None:
        # Calcular rectángulo delimitador (x, y, w, h)
        x_patente, y_patente, w_patente, h_patente = cv2.boundingRect(puntos_blancos)
        
        padding = 5
        x_final = max(0, x_patente - padding)
        y_final = max(0, y_patente - padding)
        x_end = min(img_color.shape[1], x_patente + w_patente + padding)
        y_end = min(img_color.shape[0], y_patente + h_patente + padding)

        # Dibujar BBox Azul sobre la imagen original
        img_con_bbox_patente = img_color.copy() # Copia para dibujar
        cv2.rectangle(img_con_bbox_patente, (x_final, y_final), (x_end, y_end), (255, 0, 0), 2)
        
        # 5. Recortar la Patente Completa y Guardarla
        patente_recortada = img_color[y_final:y_end, x_final:x_end]
        recortes_patentes_completas.append(patente_recortada)

        # 6. Bounding Box a Cada Carácter
        img_con_bbox_caracteres = img_color.copy() # Otra copia para dibujar BBoxes de caracteres
        
        # Recorremos la lista de BBoxes de caracteres que guardamos en la Fase 3
        caracteres_bouding_boxes = bouding_boxes_validos[i]

        for x, y, w, h in caracteres_bouding_boxes:
            # 6.1. Dibujar BBox Verde sobre la imagen original
            # Se puede agregar padding si es necesario, pero usaremos el bbox del stat (x,y,w,h)
            cv2.rectangle(img_con_bbox_caracteres, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # 7. Recortar Cada Carácter y Guardarlo
            caracter_recortado = img_color[y:y + h, x:x + w]
            
        # 7.1. Visualización de la Patente Recortada y BBoxes de Caracteres
        
        # Visualización de Bounding Box de la Patente Completa
        plt.figure()
        plt.title(f"Img {i+1}: Fase 4 - BBox Patente Completa")
        plt.imshow(cv2.cvtColor(img_con_bbox_patente, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show(block=False)

        # Visualización de Bounding Box de Cada Carácter
        plt.figure()
        plt.title(f"Img {i+1}: Fase 6 - BBox de Caracteres Válidos")
        plt.imshow(cv2.cvtColor(img_con_bbox_caracteres, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show(block=False)
        
        # Visualización de Recorte de Patente
        plt.figure()
        plt.title(f"Img {i+1}: Fase 5 - Patente Recortada")
        plt.imshow(cv2.cvtColor(patente_recortada, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show(block=False)
        
    else:
        print(f"No se detectó la patente en img{i+1}.png")
