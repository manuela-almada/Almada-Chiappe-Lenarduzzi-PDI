# 🖼️ Procesamiento de Imágenes
## Trabajo Práctico 3

### 📋 Descripción general
El presente trabajo aborda **dos problemas principales** utilizando técnicas de procesamiento digital de imágenes en **Python**:

1. **Detección automática de frames:**  
   Detectamos en 4 videos de tiradas de dados aquel frame donde los dados se encuentren detenidos.
   Luego se contabilizan los números de cada dado y se informan los resultados obtenidos.

3. **Generación de videos :**
   Se generaron 4 nuevos videos, correspondientes a cada uno de los videos originales donde los dados, mientras están en reposo, aparecen con su bounding box asociado, un nombre identificatorio y el valor obtenido.
   
---

### ⚙️ Requisitos e instalación

#### 1. Instalación de Python
Si aún no lo tenés instalado, ejecutá desde la terminal:

pip install python

#### 2. Creación de entorno virtual (opcional, pero recomendado). Para aislar las dependencias del proyecto

python -m venv venv
.\venv\Scripts\Activate.ps1

Verificá que el entorno esté activado: deberías ver (venv) al comienzo de la línea de comandos.

#### 3.Instalación de librerías necesarias. Ejecutá:

pip install matplotlib
pip install numpy
pip install opencv-contrib-python

### ▶️ Ejecución del proyecto

#### 1.Abrí el entorno de trabajo en tu editor o terminal.

#### 2.Asegurate de que las imágenes de entrada estén en la misma carpeta que tu script (tirada<id>.mp4)

#### 3.Ejecutá el scripts:
python Tp3_PDI

#### 4. Los resultados se visualizarán mediante matplotlib o por terminal.

### 📊 Resultados esperados

Problema 1: 
Se obtiene un frame del video en el que los dados se encuentran detenidos.
Sobre dicho frame se detecta cada dado de manera individual, generando un bounding box para cada uno de ellos.

A partir de cada región delimitada:

-Se identifica el número correspondiente a cada dado.

-Se muestra por terminal el valor individual de cada dado.

-Finalmente, se calcula e imprime la suma total de todos los valores detectados.

Problema 2:

-Verá un mensaje de confirmación de que el video efectivamente se generó.

-Para ver el resultado final para cada tirada deberá ir a su gestor de archivos y buscarlo (tirada_<id>_resultado.mp4) en la misma carpeta donde se encuentre alojado el script que se le provee (Tp3_PDI.py)

### 👨🏻‍💻🧑🏻‍💻👩🏻‍💻 Autores

Maximiliano Chiappe, Juan Lenarduzzi, Manuela Almada 
Cátedra: Procesamiento de Imágenes
Año: 2025
