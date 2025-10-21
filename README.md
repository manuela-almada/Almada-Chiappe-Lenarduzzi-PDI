# 🖼️ Procesamiento de Imágenes
## Trabajo Práctico 1

### 📋 Descripción general
El presente trabajo aborda **dos problemas principales** utilizando técnicas de procesamiento digital de imágenes en **Python**:

1. **Imagen con detalles escondidos:**  
   A partir de una imagen con bajo contraste, se aplican métodos de **ecualización local** para mejorar su visibilidad y revelar los detalles ocultos.

2. **Validación de formularios:**  
   Mediante **umbralado**, **detección de componentes conectadas** y análisis de caracteres, se determina si un formulario cumple las siguientes condiciones:

   - **Nombre y Apellido:** debe contener al menos dos palabras y no más de 25 caracteres.  
   - **Edad:** debe contener 2 o 3 caracteres consecutivos, sin espacios.  
   - **Mail:** debe contener una palabra y no más de 25 caracteres.  
   - **Legajo:** debe tener exactamente 8 caracteres formando una sola palabra.  
   - **Preguntas 1, 2 y 3:** en cada una debe haber **una única celda marcada** (Sí o No).  
   - **Comentarios:** debe contener al menos una palabra y no más de 25 caracteres.

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

#### 2.Asegurate de que las imágenes de entrada estén en la misma carpeta que tu script

#### 3.Ejecutá el scripts:
python Tp_Pdi1

#### 4. Los resultados se visualizarán mediante matplotlib o se guardarán en archivos de salida según el caso.

### 📊 Resultados esperados

Problema 1: Visualización clara de los detalles previamente ocultos en la imagen original.

Problema 2: Determinación automática de si cada formulario es válido o no, dicha información se ve en un archivo csv dónde almacenamos los resultados de cada validación. Y a su vez verá una imagen que informe aquellas personas que han
completado correctamente el formulario y aquellas personas que lo han completado de
forma incorrecta.

### 👨🏻‍💻🧑🏻‍💻👩🏻‍💻 Autores

Maximiliano Chiappe, Juan Lenarduzzi, Manuela Almada 
Cátedra: Procesamiento de Imágenes
Año: 2025
