# 🎯 Sistema de Detección y Seguimiento de Objetos en Tiempo Real
## Parcial - Visual Computing

### 👨‍🎓 Datos del Estudiante

| Campo | Información |
|-------|-------------|
| **Nombre completo** | Juan Jose Medina Guerrero |
| **Número de documento** | 1029980718 |
| **Correo institucional** | jmedinagu@unal.edu.co |

---

## 📋 Contenido del Informe

## 1. 🌟 Introducción del Problema y Contexto

En el ámbito de la visión por computador, la detección y seguimiento de objetos en tiempo real representa uno de los desafíos más complejos y relevantes para aplicaciones modernas. Este proyecto aborda la necesidad de crear un sistema integral que no solo detecte objetos en secuencias de video, sino que también:

- **Mantenga la identidad de los objetos** a través del tiempo (tracking)
- **Calcule métricas físicas** como velocidad y distancia recorrida
- **Proporcione visualización en tiempo real** con información enriquecida
- **Genere documentación visual** automática del proceso

El contexto de aplicación incluye:
- **Análisis de tráfico vehicular** para estudios de movilidad urbana
- **Análisis deportivo** para métricas de rendimiento de atletas
- **Sistemas de vigilancia** para detección de actividades
- **Investigación académica** en comportamiento de multitudes

## 2. 🎯 Justificación de la Solución

### Problemática Actual
Los sistemas existentes de detección de objetos típicamente trabajan de forma aislada frame por frame, sin mantener consistencia temporal. Esto resulta en:
- **Pérdida de identidad** de objetos entre frames
- **Impossibilidad de calcular métricas temporales** como velocidad
- **Falta de contexto histórico** para análisis avanzado
- **Visualización limitada** que no aprovecha información temporal

### Solución Propuesta
Nuestro sistema integra múltiples técnicas avanzadas en un pipeline coherente que:

1. **Detección robusta** con modelos YOLO de última generación
2. **Seguimiento consistente** mediante algoritmos DeepSORT
3. **Cálculo de métricas físicas** con calibración espacial
4. **Visualización enriquecida** con trayectorias e información contextual
5. **Documentación automática** con generación de GIFs y grabaciones

### Ventajas Competitivas
- **Integración completa** de múltiples técnicas en un solo sistema
- **Configuración flexible** mediante archivos YAML
- **Interfaz de usuario intuitiva** con controles en tiempo real
- **Escalabilidad** desde CPU hasta implementaciones GPU
- **Documentación automática** del proceso de análisis

## 3. 🧩 Talleres Utilizados y su Rol

| Taller | Técnica / Herramienta | Rol en la solución |
|--------|----------------------|-------------------|
| **Taller 1: Detección con YOLOv5/YOLOv8** | Modelo de detección en imágenes | Detectar objetos en cada frame con alta precisión y velocidad |
| **Taller 3: Seguimiento (tracking)** | Tracker (OSNet / DeepSORT) | Asignar ID único y mantener trayectoria consistente de cada objeto |
| **Taller 4: Cálculo de velocidad/distancia** | Transformación geométrica + métricas | Convertir píxeles a distancia real y calcular velocidad instantánea |
| **Taller 5: Visualización en tiempo real** | OpenCV + GUI (PySimpleGUI) | Mostrar bounding boxes, trayectorias e información en tiempo real |
| **Taller 7: Generación de GIFs / exportación** | Captura de GIFs del proceso | Generar evidencia visual y documentación automática del informe |

### Integración de Talleres

El sistema implementa un pipeline integrado donde cada taller aporta una capacidad específica:

```
Input → Taller 1 (Detección) → Taller 3 (Tracking) → Taller 4 (Métricas) → Taller 5 (Visualización) → Taller 7 (Documentación)
```

## 4. 📊 Diagrama de Funcionamiento

### Arquitectura General del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Video Input   │────│   FrameReader    │────│     Detector        │
│ (Cámara/Archivo)│    │   (OpenCV)       │    │    (YOLOv8)         │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Output & GIFs   │◄───│     Recorder     │◄───│      Tracker        │
│   (Archivos)    │    │   (ImageIO)      │    │    (DeepSORT)       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
          ▲                        ▲                        │
          │                        │                        ▼
          │              ┌──────────────────┐    ┌─────────────────────┐
          └──────────────│   Visualizer     │◄───│ MetricsCalculator   │
                        │    (OpenCV)      │    │   (NumPy/SciPy)     │
                        └──────────────────┘    └─────────────────────┘
```

### Flujo de Datos Detallado

1. **Entrada de Video**: Captura desde cámara o archivo
2. **Lectura de Frames**: Procesamiento secuencial con OpenCV
3. **Detección de Objetos**: Análisis con YOLOv8 para identificar objetos
4. **Seguimiento**: DeepSORT asigna IDs y mantiene trayectorias
5. **Cálculo de Métricas**: Conversión píxeles→metros, velocidad, distancia
6. **Visualización**: Renderizado en tiempo real con información enriquecida
7. **Grabación**: Captura automática de escenas interesantes y GIFs

## 5. 🎬 Evidencia de Funcionamiento

### GIFs Animados Obligatorios

> **Nota**: Los GIFs se generan automáticamente durante la ejecución del sistema y se almacenan en la carpeta `output/`.

#### Funcionalidades Demostradas:

1. **Detección y Tracking en Tiempo Real**
   - `tracking_gif_YYYYMMDD_HHMMSS.gif`
   - Muestra bounding boxes con IDs consistentes

2. **Cálculo de Métricas Visuales**
   - `metrics_gif_YYYYMMDD_HHMMSS.gif`
   - Visualización de velocidad y distancia en tiempo real

3. **Trayectorias Completas**
   - `trajectory_gif_YYYYMMDD_HHMMSS.gif`
   - Rastro histórico del movimiento de objetos

4. **Sesión Completa**
   - `session_final.gif`
   - Resumen automático de toda la sesión de análisis

### Videos Generados

El sistema genera automáticamente diversos tipos de archivos de video y GIFs durante su funcionamiento:

#### 📁 Ejemplos de Videos Generados (carpeta `output/`)

- **Sesión Final Completa**: 
  - `session_final.gif` (18MB) - Resumen completo de toda la sesión de análisis

![Sesión Final Completa](output/session_final.gif)

- **Grabaciones Automáticas**:
  - `auto_recording_20250712_193428.gif` (5.0MB) - Grabación automática de escena interesante

![Grabación Automática 1](output/auto_recording_20250712_193428.gif)

  - `auto_recording_20250712_193301.gif` (6.0MB) - Detección automática de múltiples objetos  

![Grabación Automática 2](output/auto_recording_20250712_193301.gif)

  - `auto_recording_20250712_193047.gif` (5.2MB) - Seguimiento de objetos en movimiento

![Grabación Automática 3](output/auto_recording_20250712_193047.gif)

- **Grabaciones Personalizadas**:
  - `custom_gif_20250712_193135.gif` (25MB) - GIF personalizado con configuración específica

![GIF Personalizado](output/custom_gif_20250712_193135.gif)

#### 📋 Tipos de Archivos Generados

- **Sesiones completas**: `session_YYYYMMDD_HHMMSS.mp4` / `session_final.gif`
- **Grabaciones automáticas**: `auto_recording_YYYYMMDD_HHMMSS.gif`
- **Grabaciones manuales**: `manual_HHMMSS.mp4`
- **GIFs personalizados**: `custom_gif_YYYYMMDD_HHMMSS.gif`

## 6. 🎥 Enlace al Video

> **📹 Video Demostrativo del Sistema**
> 
> **Enlace**: https://drive.google.com/file/d/1vtydqual_R-3nlUkeiy6tOEBUzqOAG5X/view?usp=sharing


> 
> El video muestra:
> - ✅ Funcionamiento completo del sistema en tiempo real
> - ✅ Detección y seguimiento de múltiples objetos
> - ✅ Cálculo de métricas de velocidad y distancia
> - ✅ Visualización de trayectorias e información contextual
> - ✅ Interfaz gráfica y controles del usuario
> - ✅ Generación automática de GIFs y grabaciones

## 7. 🔧 Explicación Técnica del Funcionamiento

### Componentes Principales

#### 7.1 FrameReader (Lectura de Video)
```python
class FrameReader:
    def __init__(self, source, frame_width=None, frame_height=None):
        # Inicialización de captura de video con OpenCV
        # Soporte para cámaras y archivos de video
```

**Características**:
- Soporte multi-fuente (cámara, archivos, streams)
- Redimensionamiento automático de frames
- Control de FPS y buffer de frames

#### 7.2 Detector (YOLOv8)
```python
class Detector:
    def __init__(self, model_path, confidence_threshold=0.5, device="cpu"):
        # Carga del modelo YOLOv8 pre-entrenado
        # Configuración de umbrales y dispositivo de cómputo
```

**Características**:
- Modelos YOLOv8 (nano, small, medium, large, xlarge)
- Filtrado por clases específicas
- Optimización GPU/CPU automática
- NMS (Non-Maximum Suppression) configurable

#### 7.3 Tracker (DeepSORT)
```python
class Tracker:
    def __init__(self, max_age=50, min_hits=3, iou_threshold=0.3):
        # Inicialización del tracker DeepSORT
        # Configuración de parámetros de asociación
```

**Características**:
- **Kalman Filter**: Predicción de movimiento
- **Hungarian Algorithm**: Asociación de detecciones
- **Deep Features**: Características visuales para re-identificación
- **Track Management**: Gestión del ciclo de vida de tracks

#### 7.4 MetricsCalculator (Cálculo de Métricas)
```python
class MetricsCalculator:
    def __init__(self, pixels_per_meter=50, fps=30):
        # Calibración espacial y temporal
        # Inicialización de historial de métricas
```

**Métricas Implementadas**:
- **Velocidad Instantánea**: `v = Δposition / Δtime`
- **Velocidad Suavizada**: Media móvil en ventana temporal
- **Distancia Acumulada**: Integral de velocidad en el tiempo
- **Aceleración**: `a = Δvelocity / Δtime`
- **Índice de Suavidad**: Consistencia de la trayectoria

#### 7.5 Visualizer (Renderizado)
```python
class Visualizer:
    def __init__(self, show_trajectory=True, trajectory_length=30):
        # Configuración de elementos visuales
        # Inicialización de colores y estilos
```

**Elementos Visuales**:
- **Bounding Boxes**: Rectángulos con colores por clase
- **IDs de Tracks**: Identificadores únicos persistentes
- **Trayectorias**: Historial de posiciones con fade
- **Métricas en Tiempo Real**: Velocidad, distancia, aceleración
- **Estadísticas Globales**: FPS, objetos activos, resumen

#### 7.6 Recorder (Grabación)
```python
class Recorder:
    def __init__(self, output_dir="output", auto_record=True):
        # Configuración de grabación automática y manual
        # Gestión de buffers y formatos de salida
```

**Capacidades de Grabación**:
- **Auto-grabación**: Detección de escenas interesantes
- **Grabación Manual**: Control por usuario (tecla R)
- **GIFs Automáticos**: Buffer circular de frames recientes
- **Múltiples Formatos**: MP4, AVI, GIF

### Sistema de Configuración

El sistema utiliza archivos YAML para configuración flexible:

```yaml
# config/config.yaml
detector:
  model_path: "models/yolov8n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: "cpu"
  classes_to_detect: []

tracker:
  max_age: 50
  min_hits: 3
  iou_threshold: 0.3

metrics:
  pixels_per_meter: 50
  fps: 30
  velocity_smoothing_window: 10

visualizer:
  show_trajectory: true
  trajectory_length: 30
  show_velocity: true
  show_distance: true

recorder:
  auto_record_interesting_scenes: true
  gif_duration: 5.0
  buffer_size: 150
```

### Pipeline de Procesamiento

1. **Inicialización**:
   ```python
   frame_reader = FrameReader(source)
   detector = Detector(config.detector)
   tracker = Tracker(config.tracker)
   metrics_calculator = MetricsCalculator(config.metrics)
   visualizer = Visualizer(config.visualizer)
   recorder = Recorder(config.recorder)
   ```

2. **Loop Principal**:
   ```python
   while True:
       frame = frame_reader.read_frame()
       detections = detector.detect(frame)
       tracks = tracker.update(detections)
       metrics = metrics_calculator.update(tracks)
       visualized_frame = visualizer.draw(frame, tracks, metrics)
       recorder.process_frame(visualized_frame)
   ```

3. **Gestión de Estados**:
   - Control de pausa/reanudación
   - Grabación manual y automática
   - Generación de GIFs bajo demanda
   - Exportación de métricas

## 8. 🚀 Instalación y Uso

### Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd parcial-visual

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### Uso

#### Modo Terminal
```bash
# Cámara por defecto
python main.py

# Archivo de video
python main.py --source video.mp4

# Configuración personalizada
python main.py --confidence 0.7 --device cuda
```

#### Modo GUI
```bash
python gui_app.py
```

#### Controles
- **Q**: Salir
- **ESPACIO**: Pausar/Reanudar
- **R**: Grabación manual
- **G**: Crear GIF
- **S**: Estadísticas

## 9. 🏆 Conclusiones y Reflexiones Personales

### Logros Técnicos Alcanzados

1. **Integración Exitosa de Múltiples Técnicas**
   - La combinación de YOLO + DeepSORT + métricas físicas resultó en un sistema robusto y funcional
   - El pipeline de procesamiento mantiene rendimiento en tiempo real incluso con múltiples objetos

2. **Flexibilidad y Configurabilidad**
   - El sistema de configuración YAML permite adaptación rápida a diferentes escenarios
   - La modularidad del código facilita la extensión y modificación de componentes

3. **Interfaz de Usuario Intuitiva**
   - La GUI desarrollada con PySimpleGUI proporciona control completo sin conocimiento técnico
   - Los controles de teclado permiten operación eficiente durante el análisis

### Desafíos Superados

1. **Calibración Espacial**
   - **Problema**: Conversión precisa de píxeles a unidades métricas reales
   - **Solución**: Sistema de calibración flexible con parámetro `pixels_per_meter` ajustable

2. **Rendimiento en Tiempo Real**
   - **Problema**: Mantener FPS estable con múltiples algoritmos complejos
   - **Solución**: Optimización del pipeline y configuración adaptativa de parámetros

3. **Persistencia de IDs**
   - **Problema**: Pérdida de identidad de objetos en oclusiones o salidas temporales
   - **Solución**: Configuración óptima de DeepSORT y gestión inteligente de tracks

### Aprendizajes Clave

1. **Importancia de la Modularidad**
   - El diseño modular permitió desarrollo, testing y debugging independiente de cada componente
   - Facilita futuras extensiones y mantenimiento del código

2. **Configuración vs. Hardcoding**
   - El uso de archivos de configuración externa mejora dramáticamente la usabilidad
   - Permite experimentación rápida sin modificar código

3. **Visualización como Herramienta de Debug**
   - La visualización rica en tiempo real es invaluable para detectar problemas en los algoritmos
   - Los GIFs automáticos proporcionan documentación visual del comportamiento del sistema

### Aplicaciones Futuras

1. **Análisis de Tráfico Urbano**
   - Estudios de flujo vehicular para planificación urbana
   - Detección de patrones de comportamiento en intersecciones

2. **Deportes y Entrenamientos**
   - Análisis de rendimiento de atletas con métricas objetivas
   - Seguimiento de tácticas de equipo en deportes colectivos

3. **Seguridad y Vigilancia**
   - Detección automática de comportamientos anómalos
   - Análisis de multitudes en eventos masivos

### Reflexiones sobre Visual Computing

Este proyecto me ha permitido comprender la complejidad y potencial de los sistemas de visión por computador modernos. La integración de múltiples técnicas no es trivial y requiere:

- **Comprensión profunda** de cada algoritmo y sus limitaciones
- **Habilidades de ingeniería de software** para crear sistemas mantenibles
- **Pensamiento sistémico** para optimizar el rendimiento global
- **Atención al detalle** en la experiencia de usuario

### Impacto Personal y Académico

1. **Crecimiento Técnico**
   - Dominio práctico de OpenCV, YOLO, y algoritmos de tracking
   - Experiencia en optimización de performance y gestión de memoria
   - Desarrollo de interfaces de usuario efectivas

2. **Metodología de Trabajo**
   - Importancia de la documentación continua y clara
   - Valor del testing iterativo con datos reales
   - Beneficios del desarrollo modular y configuración externa

3. **Visión de Futuro**
   - Interés en explorar técnicas más avanzadas de deep learning para tracking
   - Motivación para aplicar estos conocimientos en proyectos de investigación
   - Comprensión del potencial de la visión por computador en aplicaciones reales
---

## 📁 Estructura del Proyecto

```
parcial-visual/
├── src/
│   ├── core/                    # Módulos principales
│   │   ├── frame_reader.py      # Lectura de video
│   │   ├── detector.py          # Detección YOLO
│   │   ├── tracker.py           # Seguimiento DeepSORT
│   │   ├── metrics_calculator.py # Cálculo de métricas
│   │   ├── visualizer.py        # Renderizado
│   │   └── recorder.py          # Grabación
│   └── utils/
│       └── config_loader.py     # Gestión de configuración
├── config/
│   └── config.yaml              # Configuración principal
├── models/                      # Modelos pre-entrenados
├── data/                        # Videos de entrada
├── output/                      # Resultados generados
├── main.py                      # Aplicación principal
├── gui_app.py                   # Interfaz gráfica
├── requirements.txt             # Dependencias
└── README.md                    # Esta documentación
```

