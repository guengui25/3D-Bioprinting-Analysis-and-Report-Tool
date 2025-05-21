# Herramienta de Análisis y Reporte de Bioimpresión 3D

Una aplicación integral para analizar estructuras 3D bioimpresas mediante el procesamiento de G‑code y análisis de imágenes. Esta herramienta ayuda a investigadores e ingenieros a evaluar la calidad de impresión, medir anchos de línea, comparar diseños con resultados reales y generar informes detallados.

## Tabla de contenidos
- [Visión general](#visión-general)
- [Funciones clave](#funciones-clave)
- [Requisitos del sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Procesamiento de G‑code](#procesamiento-de-g-code)
  - [Análisis de imágenes](#análisis-de-imágenes)
  - [Trabajo con informes](#trabajo-con-informes)
- [Flujo de trabajo](#flujo-de-trabajo)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Detalles técnicos](#detalles-técnicos)
- [Limitaciones](#limitaciones)

## Visión general
La **Herramienta de Análisis y Reporte de Bioimpresión 3D** proporciona un entorno integrado para evaluar la calidad de las estructuras bioimpresas comparando las instrucciones originales del G‑code con los resultados impresos. La aplicación procesa archivos G‑code para extraer trayectorias de impresión, analiza fotografías de los objetos impresos, superpone las trayectorias previstas sobre los resultados reales y genera informes exhaustivos con métricas detalladas sobre la precisión de la impresión.

> **Nota:** Actualmente, la herramienta solo admite archivos G‑code generados por **PrusaSlicer**.

## Funciones clave

### Procesamiento de G‑code
- **Análisis de archivos G‑code**: Extrae trayectorias de impresión e información de capas.
- **Detección inteligente del perímetro**: Identifica automáticamente y, opcionalmente, filtra el perímetro exterior de las impresiones.
- **Visualización capa por capa**: Crea visualizaciones de cada capa con la línea actual resaltada.
- **Detección de escala**: Determina automáticamente el factor de escala según el diámetro de la boquilla.
- **Generación secuencial de imágenes**: Crea secuencias tipo *time‑lapse* que muestran el proceso de impresión.

### Procesamiento de imágenes
- **Detección automática de escala**: Identifica información de escala a partir de elementos de referencia en las fotografías.
- **Corrección de distorsión**: Detecta y compensa la distorsión de la lente de la cámara.
- **Segmentación de figura 3D**: Extrae automáticamente el objeto impreso del fondo.
- **Superposición de G‑code**: Mapea con precisión las trayectorias de G‑code sobre la imagen procesada.
- **Generación de GIF de acumulación**: Crea visualizaciones animadas del proceso de impresión.

### Análisis y métricas
- **Medición del ancho de línea**: Calcula mediciones precisas de los anchos de línea impresos.
- **Análisis de variación de ancho**: Determina la consistencia del grosor de línea a lo largo de la impresión.
- **Detección de errores**: Identifica líneas que cruzan contornos o presentan anchos irregulares.
- **Análisis estadístico**: Genera estadísticas completas sobre la precisión de la impresión.
- **Detección de contornos**: Identifica contornos externos e internos para análisis dimensional.
- **Análisis basado en muestras**: Permite configurar el número de puntos de muestra para mediciones detalladas.

### Generación de informes
- **Informes PDF integrales**: Genera informes detallados que incluyen todas las métricas de análisis.
- **Imágenes de visualización**: Incluye imágenes procesadas que muestran los resultados del análisis.
- **Tablas comparativas**: Presenta mediciones exactas junto con los valores esperados.
- **Resaltado de errores**: Identifica visualmente las áreas problemáticas con códigos de color.
- **Apéndices con detalles de líneas**: Incluye análisis de líneas individuales con métricas.

### Plantillas imprimibles
- **Plantillas de referencia**: Genera plantillas PDF imprimibles con elementos de calibración.
- **Referencias de escala**: Incluye reglas y tarjetas de referencia precisas para detección de escala.
- **Múltiples variantes**: Versiones con fondo blanco y negro para diferentes materiales.
- **Guía de fotografía optimizada**: Instrucciones para la correcta colocación de la cámara.

> **Consulta la** [Documentación de plantillas imprimibles](docs/printable_templates.md)

### Interfaz de usuario
- **Interfaz intuitiva con pestañas**: Pestañas separadas para procesamiento de G‑code y análisis de imágenes.
- **Salida de consola integrada**: Muestra la información de procesamiento directamente en la aplicación.
- **Selección de archivos y directorios**: Permite explorar archivos mediante cuadros de diálogo.
- **Opciones de procesamiento configurables**: Permite personalizar los parámetros de análisis.
- **Procesamiento concurrente**: Procesa en segundo plano con indicadores de progreso.
- **Monitorización de estado**: Actualizaciones en tiempo real durante el análisis.

## Requisitos del sistema
- **Sistema operativo**: Windows, macOS o Linux.
- **Python**: Python 3.12.10.
- **Espacio en disco**: Mínimo 900 MB para la instalación y espacio adicional para los archivos resultantes.
- **RAM**: Mínimo 4 GB (se recomiendan 8 GB o más para archivos grandes).
- **Software requerido**: [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) (para OCR en la detección de texto de escala).

## Instalación
1. Clona o descarga el repositorio:
   ```bash
   git clone <repository-url>
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd "3D Bioprinting Analysis and Report Tool"
   ```
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
4. Asegúrate de que **Tesseract OCR** esté instalado en tu sistema:
   - **Windows**: Descarga e instala desde [GitHub](https://github.com/tesseract-ocr/tesseract).
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`
5. Ejecuta la aplicación:
   ```bash
   python run_app.py
   ```

## Uso

### Procesamiento de G‑code
1. En la pestaña **“G‑Code Processing”**, haz clic en **“Browse…”** y selecciona un archivo G‑code.
2. Especifica un directorio de salida o utiliza el predeterminado.
3. Configura las opciones de procesamiento:
   - Activa **“Skip Perimeter”** para filtrar el perímetro exterior (recomendado).
   - Ajusta **“Perimeter %”** para controlar cuánto de los segmentos iniciales se filtrará.
   - Activa **“Limit Layers”** y define **“Max Layers”** para procesar solo capas específicas.
4. Haz clic en **“Process G‑Code”** para iniciar el análisis.
5. Observa el progreso y los resultados en la consola.
6. Al finalizar, el directorio de G‑code se establecerá automáticamente para el análisis de imágenes.

### Análisis de imágenes
1. En la pestaña **“Image Processing”**, haz clic en **“Browse…”** y selecciona una imagen de tu impresión 3D.
2. Usa el directorio de imágenes de G‑code que se rellena automáticamente o selecciona otro.
3. Especifica un directorio de salida para los resultados del análisis.
4. Configura las opciones de procesamiento:
   - Activa **“Save Analysis Images”** para guardar visualizaciones individuales.
   - Activa o desactiva **“Generate GIF”** para crear una animación del proceso.
   - Activa **“Generate PDF Report”** si deseas un informe detallado.
   - Configura **“Number of samples”** para definir los puntos de medición de ancho.
5. Haz clic en **“Process Image”** para comenzar el análisis.
6. Supervisa el progreso en la consola.
7. Al finalizar, se mostrará un mensaje de éxito con la ruta de salida.

### Trabajo con informes
1. Navega al directorio de salida especificado durante el análisis de imágenes.
2. Abre el archivo PDF (`analysis_report_*.pdf`) para ver los resultados detallados.
3. El informe contiene:
   - Información general de la impresión y el análisis.
   - Resumen estadístico de los anchos de línea.
   - Visualización de los pasos de procesamiento.
   - Tablas detalladas de medición de anchos.
   - Apéndice con imágenes de análisis individuales.
4. También encontrarás archivos CSV con los datos originales en la carpeta `line_analysis`.

## Flujo de trabajo
1. **Prepara la impresión**: Imprime tu estructura 3D.
2. **Captura la imagen**: Fotografía el resultado usando la [plantilla imprimible](#plantillas-imprimibles) como referencia.
3. **Procesa el G‑code**: Extrae las trayectorias y crea una representación visual.
4. **Analiza la imagen**: Procesa la fotografía para detectar y segmentar la estructura.
5. **Genera la superposición**: Mapea las trayectorias del G‑code en la imagen procesada.
6. **Analiza los resultados**: Calcula métricas sobre anchos de línea y precisión.
7. **Revisa el informe**: Examina el PDF para estudiar los resultados en detalle.

> **Ejemplo de salida** disponible [aquí](aux/sample_data/output_sample)

## Estructura del proyecto
```text
3D Bioprinting Analysis and Report Tool/
├── app/                          # Código principal de la aplicación
│   ├── functions/                  # Módulos principales
│   │   ├── figure_detection/       # Segmentación de objetos bimpresos en 3D
│   │   │   └── figure_detection.py       # Segmentación de objetos
│   │   ├── fix_distorsion/         # Corrección de distorsión
│   │   │   └── fix_distorsion.py         # Corrección de distorsión
│   │   ├── overlay_gcode/          # Procesamiento de G‑code
│   │   │   ├── overlay_lines.py          # Superponer G‑code en imágenes
│   │   │   └── scale_gcode.py            # Extracción escala G‑code
│   │   ├── reporting/              # Generación de informes
│   │   │   └── pdf_generator.py          # Crear informes en PDF
│   │   ├── scale_detection/           # Detección de escala
│   │   │   └── scale_ruler.py            # Detección de escala de la regla
│   │   └── width_detection/           # Detecta anchos de línea y genera métricas
│   │       ├── analyze_lines.py          # Medición de anchos de línea
│   │       └── metrics_exporter.py       # Exportar métricas
│   ├── app_gui.py                  # Interfaz gráfica
│   ├── main_gcode.py               # Flujo de G‑code
│   └── main_mask.py                # Flujo de imágenes
├── aux/                          # Archivos auxiliares
│   ├── sample_data/                # Datos de ejemplo
│   │   ├── input_sample/              # Entradas de ejemplo
│   │   └── output_sample/             # Salidas de ejemplo
│   └── template_generation/        # Generación de plantillas
│       └── printable_template.py      # Crear plantillas
├── docs/                         # Documentación
│   ├── printable_templates.md      # Detalles de plantillas
│   └── README_es.md                # README en castellano
├── logs/                         # Registros de la aplicación
├── templates/                    # Plantillas de impresión
│   ├── printable_template_black.pdf 
│   └── printable_template_white.pdf
├── .gitignore
├── README.md                     # README original
├── requirements.txt              # Dependencias
└── run_app.py                    # Punto de entrada
```

## Detalles técnicos
### Arquitectura
La aplicación es modular e incluye:
- **Analizador de G‑code**: Extrae trayectorias e información de capas (compatible con PrusaSlicer).
- **Procesador de imágenes**: Corrección, segmentación y análisis de fotografías.
- **Motor de superposición**: Mapea trayectorias de G‑code a coordenadas de imagen.
- **Módulo de análisis**: Calcula anchos de línea y métricas de calidad.
- **Generador de informes**: Produce informes PDF con resultados.
- **Interfaz GUI**: Permite acceder a todas las funciones desde la aplicación.

### Archivos clave
- `run_app.py`: Punto de inicio de la aplicación.
- `app_gui.py`: Implementación de la interfaz.
- `main_gcode.py`: Procesamiento de G‑code.
- `main_mask.py`: Procesamiento de imágenes.
- `printable_template.py`: Generación de plantillas de referencia.

## Limitaciones
- Solo admite archivos G‑code generados por PrusaSlicer.
- La detección de escala requiere reglas o marcadores rojos en la imagen.
- Los mejores resultados se obtienen usando las plantillas proporcionadas.
- El rendimiento puede degradarse con archivos G‑code muy grandes o imágenes de alta resolución.
