# 🖼️ Procesamiento de Imágenes
## Trabajo Práctico 2

### 📋 Descripción general
El presente trabajo aborda **dos problemas principales** utilizando técnicas de procesamiento digital de imágenes en **Python**:

1. **Detección de monedas y dados:**  
   A partir de una imagen con monedas de distintos tipos y dados, se aplican métodos de mofología y segmentación para identificar ambos elementos y luego clasificar y contar las monedas y contar los puntos presentes en las caras superiores de los dados

2. **Detección de patentes :**
   En 12 imagenes de automóviles:
   Detectamos automáticamente la placa patente y segmentamos la misma.
   Implementamos un algoritmo de procesamiento que segmenta los caracteres de la placa patente detectada.
   
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

#### 2.Asegurate de que las imágenes de entrada estén en la misma carpeta que tu script (img01.png - img12.png y monedas.jpg)

#### 3.Ejecutá el scripts:
python Tp2_PDI

#### 4. Los resultados se visualizarán mediante matplotlib o por terminal.

### 📊 Resultados esperados

Problema 1: 
-Imagen original donde los dados se ven con un contorno rojo y las monedas con un contorno verde tras la detección automática
-Cantidad de monedas y cantidad de puntos de los dados en terminal.
-Observará ciertas imagenes a lo largo de la ejecución que  demuestran un paso a paso de como se modifican las imagenes hasta llegar a los resultados finales.

Problema 2: 
-Imagenes con un bounding box en la patente (en varias imagenes no se logró la detección)
-Imágenes con un boundingbox para cada caracter de la patente.
-Observará ciertas imagenes a lo largo de la ejecución que  demuestran un paso a paso de como se modifican las imagenes hasta llegar a los resultados finales.

### 👨🏻‍💻🧑🏻‍💻👩🏻‍💻 Autores

Maximiliano Chiappe, Juan Lenarduzzi, Manuela Almada 
Cátedra: Procesamiento de Imágenes
Año: 2025
