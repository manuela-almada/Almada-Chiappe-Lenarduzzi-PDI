import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- FUNCION AUXILIAR  ---
def imfillhole_v2(img_binary):
    """
    Rellena los huecos dentro de los objetos binarios.
    Equivalente a imfill(BW, 'holes') de MATLAB.
    """
    # Copiar la imagen original
    im_floodfill = img_binary.copy()
    
    # Crear una máscara (tamaño imagen + 2 pixeles en cada borde)
    h, w = img_binary.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Inundar desde el punto (0,0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invertir la imagen inundada
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combinar la imagen original con la invertida
    im_out = img_binary | im_floodfill_inv
    
    return im_out

# --- FUNCIÓN PRINCIPAL ---
def obtener_frame_dados_quietos(video_path):
    """
    Procesa el video buscando 5 dados rojos. Retorna el frame donde se quedan quietos.
    
    Retorna:
        (imagen, exito): 
            - imagen: El frame capturado (o None si falló).
            - exito: True si detectó la quietud, False si terminó el video sin éxito.
    """
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return None, False

    # --- CONSTANTES DE TU LÓGICA ---
    # Nota: Si el video cambia de resolución, width/height se ajustan solos
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    AREA_MINIMA = 250   
    AREA_MAXIMA = 900
    DADOS_ESPERADOS = 5
    UMBRAL_MOVIMIENTO = 2.0 
    FRAMES_QUIETOS_NECESARIOS = 3

    # Variables de Estado
    centroides_anteriores = None
    contador_quietud = 0
    imagen_final_capturada = None 
    detectado = False

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break # Fin del video

        # Redimensionar (Manteniendo tu lógica de width/3)
        frame_resized = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
        
        # --- PROCESAMIENTO ---
        frame_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(frame_hsv)

        # Máscara lógica para detectar rojo
        ix = np.logical_and(
            np.logical_or(np.logical_and(h > 180 * .9, h < 180), h < 180 * 0.04), 
            np.logical_and(s > 256 * 0.3, s < 256)
        )
        mask = ix.astype(np.uint8) * 255
        
        resultado_rojo = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)
        gray_rojo = cv2.cvtColor(resultado_rojo, cv2.COLOR_BGR2GRAY)
        
        # Threshold adaptativo
        thresh = cv2.adaptiveThreshold(gray_rojo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        
        # Relleno de huecos
        dados = imfillhole_v2(thresh)
        if dados.dtype == bool: dados = dados.astype(np.uint8) * 255

        # --- OBTENER CANDIDATOS VÁLIDOS ---
        contours, _ = cv2.findContours(dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centroides_actuales = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # FILTRO POR ÁREA
            if AREA_MINIMA < area < AREA_MAXIMA:
                x, y, w, h_rect = cv2.boundingRect(cnt)
                ratio = float(w) / h_rect
                
                # FILTRO POR RATIO
                if 0.7 <= ratio <= 1.3:
                    # CALCULO DE CENTROIDE (MOMENTOS)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centroides_actuales.append((cX, cY))

        # --- LÓGICA DE MOVIMIENTO DE CENTROIDES ---
        
        # 1. Solo analizamos si detectamos exactamente 5 dados
        if len(centroides_actuales) == DADOS_ESPERADOS:
            
            # 2. ORDENAR CENTROIDES (Izquierda a Derecha)
            centroides_actuales.sort(key=lambda x: x[0])

            if centroides_anteriores is not None:
                movimiento_total = 0
                
                # 3. Calcular distancia entre posiciones actuales y anteriores
                for i in range(DADOS_ESPERADOS):
                    c_act = centroides_actuales[i]
                    c_ant = centroides_anteriores[i]
                    
                    # Distancia Euclidiana
                    dist = np.sqrt((c_act[0] - c_ant[0])**2 + (c_act[1] - c_ant[1])**2)
                    movimiento_total += dist
                
                # Promedio de movimiento por dado
                movimiento_promedio = movimiento_total / DADOS_ESPERADOS
                
                # 4. Chequear si están quietos
                if movimiento_promedio < UMBRAL_MOVIMIENTO:
                    contador_quietud += 1
                else:
                    contador_quietud = 0 # Se movieron
            
            # Actualizamos para el siguiente frame
            centroides_anteriores = centroides_actuales

            # 5. Criterio de Finalización
            if contador_quietud >= FRAMES_QUIETOS_NECESARIOS:
                imagen_final_capturada = frame_resized.copy()
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                detectado = True
                break
        else:
            # Si perdemos un dado o aparece ruido, reseteamos la cuenta de quietud
            contador_quietud = 0
            centroides_anteriores = None

    cap.release()
    
    return imagen_final_capturada, detectado,frame_idx

def obtener_frame_dados_quietos_especial(video_path):
    """
    Devuelve el mejor frame donde los dados están quietos,
    aunque nunca se detecten los 5 perfectamente al mismo tiempo.

    Retorna:
        frame (imagen),
        exito (bool),
        frame_idx (int)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return None, False, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- PARÁMETROS ---
    AREA_MINIMA = 250
    AREA_MAXIMA = 900
    DADOS_MINIMOS = 3
    UMBRAL_MOVIMIENTO = 2.0

    mejor_frame = None
    mejor_score = -1
    mejor_frame_idx = None

    centroides_previos = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, dsize=(int(width/3), int(height/3)))

        # --- SEGMENTACIÓN ROJA ---
        frame_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(frame_hsv)

        ix = np.logical_and(
            np.logical_or(np.logical_and(h > 180 * .9, h < 180), h < 180 * 0.04),
            np.logical_and(s > 256 * 0.3, s < 256)
        )
        mask = ix.astype(np.uint8) * 255

        resultado_rojo = cv2.bitwise_and(frame_resized, frame_resized, mask=mask)
        gray_rojo = cv2.cvtColor(resultado_rojo, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray_rojo, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )

        dados = imfillhole_v2(thresh)
        if dados.dtype == bool:
            dados = dados.astype(np.uint8) * 255

        contours, _ = cv2.findContours(dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroides_actuales = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if AREA_MINIMA < area < AREA_MAXIMA:
                x, y, w, h_rect = cv2.boundingRect(cnt)
                ratio = float(w) / h_rect
                if 0.7 <= ratio <= 1.3:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cX = int(M['m10'] / M['m00'])
                        cY = int(M['m01'] / M['m00'])
                        centroides_actuales.append((cX, cY))

        num_dados = len(centroides_actuales)

        # --- MEDIDA DE FOCO ---
        gray_full = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        foco = cv2.Laplacian(gray_full, cv2.CV_64F).var()

        # --- MOVIMIENTO APROXIMADO ---
        movimiento = 0
        if centroides_previos is not None and num_dados >= DADOS_MINIMOS:
            c1 = sorted(centroides_previos, key=lambda x: x[0])
            c2 = sorted(centroides_actuales, key=lambda x: x[0])
            n = min(len(c1), len(c2))
            if n > 0:
                for i in range(n):
                    movimiento += np.linalg.norm(np.array(c1[i]) - np.array(c2[i]))
                movimiento /= n

        centroides_previos = centroides_actuales

        # --- SCORE GLOBAL ---
        score = num_dados * 1000 + foco - movimiento * 50

        if num_dados >= DADOS_MINIMOS and score > mejor_score:
            mejor_score = score
            mejor_frame = frame_resized.copy()
            mejor_frame_idx = frame_idx

        frame_idx += 1

    cap.release()

    if mejor_frame is not None:
        return mejor_frame, True, mejor_frame_idx
    else:
        return None, False, None




def analizar_valores_dados(imagen,es_imagen4=False):
    """
    Recibe la imagen de los dados quietos, segmenta cada dado y cuenta sus puntos.
    Estrategia mejorada: Recorte de bordes y Canny sin dilatación para baja resolución.
    """
    if imagen is None:
        print("Error: La imagen de entrada es None.")
        return

    # Usamos la variable de entrada 'imagen', no 'imagen1'
    imagen_final = imagen.copy()
    
    # 1. PREPROCESAMIENTO GLOBAL (Detectar los dados rojos)
    # -----------------------------------------------------
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    # Tu máscara HSV exacta
    ix = np.logical_and(
        np.logical_or(np.logical_and(h > 180 * .9, h < 180), h < 180 * 0.04), 
        np.logical_and(s > 256 * 0.3, s < 256)
    )
    mask = ix.astype(np.uint8) * 255

    # Limpieza inicial
    kernel_dado = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_dado, iterations=1)
    
    # Relleno de huecos (Asumiendo que tienes imfillhole_v2 definida externamente)
    # Si no la tienes, puedes usar: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    try:
        dados_solidos = imfillhole_v2(mask)
    except NameError:
        # Fallback si no existe la función
        dados_solidos = mask 

    if dados_solidos.dtype == bool: dados_solidos = dados_solidos.astype(np.uint8) * 255

    # Encontrar contornos de los DADOS
    contours_dados, _ = cv2.findContours(dados_solidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    AREA_MIN_DADO = 200    
    AREA_MAX_DADO = 5000   
    
    # AJUSTE: Bajamos el área mínima porque al no dilatar los puntos son más "finos"
    AREA_MIN_PUNTO = 2     
    AREA_MAX_PUNTO = 150    

    resultados = {}
    idx = 0

    # 2. ANALIZAR CADA DADO INDIVIDUALMENTE
    # -------------------------------------
    for cnt_dado in contours_dados:
        area_dado = cv2.contourArea(cnt_dado)

        if AREA_MIN_DADO < area_dado < AREA_MAX_DADO:
            idx += 1
            x, y, w, h_rect = cv2.boundingRect(cnt_dado)
            
            # Recorte inicial del dado completo
            roi_color_full = imagen[y:y+h_rect, x:x+w]
            
            # --- MODIFICACIÓN CLAVE: RECORTAR BORDES INTERNOS ---
            if es_imagen4:
                margin_h = int(h_rect * 0.03)
                margin_w = int(w * 0.03)
            else:
                margin_h = int(h_rect * 0.07)
                margin_w = int(w * 0.07)
            
            # Validamos que el recorte sea posible
            if h_rect > 2 * margin_h and w > 2 * margin_w:
                roi_color = roi_color_full[margin_h:h_rect-margin_h, margin_w:w-margin_w]
            else:
                roi_color = roi_color_full # Si es muy chico, no recortamos

            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            
            # --- ESTRATEGIA DE DETECCIÓN DE PUNTOS ---
            
            # A) Suavizado leve
            blur_roi = cv2.GaussianBlur(roi_gray, (3,3), 0)
        
            # C) Canny 
            edges_roi = cv2.Canny(blur_roi, 62, 132)
            
            # D) Encontrar contornos sobre los bordes FINOS
            thresh_roi = edges_roi 
            contours_puntos, _ = cv2.findContours(thresh_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            puntos_validos = []
            
            # E) FILTRAR PUNTOS
            h_roi_inner, w_roi_inner = roi_color.shape[:2]
            max_area_dinamica = (h_roi_inner * w_roi_inner) / 4 # No puede ser mas grande que 1/4 del dado

            for cnt_punto in contours_puntos:
                area_punto = cv2.contourArea(cnt_punto)
                
                # Filtro por área
                if AREA_MIN_PUNTO < area_punto < max_area_dinamica:
                    
                    # Filtro de circularidad 
                    perimetro = cv2.arcLength(cnt_punto, True)
                    if perimetro > 0:
                        circularidad = 4 * np.pi * (area_punto / (perimetro * perimetro))
                        
                        if circularidad > 0.4: 
                            puntos_validos.append(cnt_punto)

            cantidad_puntos = len(puntos_validos)
            resultados[f"Dado {idx}"] = cantidad_puntos

            # Dibujar en la imagen principal
            cv2.rectangle(imagen_final, (x, y), (x+w, y+h_rect), (255, 0, 0), 2)
            # Dibujamos un rectangulo interior para ver qué estamos analizando realmente
            cv2.rectangle(imagen_final, (x+margin_w, y+margin_h), (x+w-margin_w, y+h_rect-margin_h), (0, 255, 255), 1)
            cv2.putText(imagen_final, str(cantidad_puntos), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 3. RESULTADOS FINALES
    # ---------------------
    print("\n===============================")
    print("   RESULTADO DE LA TIRADA")
    print("===============================")
    total_suma = 0
    for key, val in resultados.items():
        print(f"{key}: {val} puntos")
        total_suma += val
    print(f"TOTAL SUMA: {total_suma}")
    
    # Mostrar imagen completa final
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(imagen_final, cv2.COLOR_BGR2RGB))
    plt.title(f"Deteccion Final - Suma: {total_suma}")
    plt.show(block=False)
    return imagen_final, resultados



def generar_video_reemplazando_frame(
    video_path,
    output_path,
    frame_reemplazo,
    frame_reemplazo_idx,
    frames_extra=15
):
    cap = cv2.VideoCapture(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    frame_reemplazo_full = cv2.resize(frame_reemplazo, (width, height))

    idx = 0
    contador_extra = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx == frame_reemplazo_idx:
            out.write(frame_reemplazo_full)
            contador_extra = frames_extra
        elif contador_extra > 0:
            out.write(frame_reemplazo_full)
            contador_extra -= 1
        else:
            out.write(frame)

        idx += 1

    cap.release()
    out.release()
    
    
    
videos = [
    "tirada_1.mp4",
    "tirada_2.mp4",
    "tirada_3.mp4",
    "tirada_4.mp4"
]

for i, video in enumerate(videos, start=1):

    print(f"\nProcesando tirada {i}...")

    # Obtener frame quieto
    if i == 2:
        frame, ok, frame_idx = obtener_frame_dados_quietos_especial(video)
    else:
        frame, ok, frame_idx = obtener_frame_dados_quietos(video)

    if not ok:
        print(f"No se pudo procesar {video}")
        continue

    # Analizar dados
    frame_anotado, resultados = analizar_valores_dados(
        frame,
        es_imagen4=(i == 4)
    )

    # Generar video final
    generar_video_reemplazando_frame(
        video_path=video,
        output_path=f"tirada_{i}_resultado.mp4",
        frame_reemplazo=frame_anotado,
        frame_reemplazo_idx=frame_idx,
        frames_extra=15
    )

    print(f"Video generado: tirada_{i}_resultado.mp4")
