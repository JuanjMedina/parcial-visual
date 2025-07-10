# 🎯 Sistema de Detección y Seguimiento de Objetos en Tiempo Real

Un sistema completo de visión por computador que integra múltiples técnicas avanzadas para la detección, seguimiento y análisis de objetos en video en tiempo real.

## 🌟 Características Principales

- **🔍 Detección con YOLO**: Detección precisa de objetos usando YOLOv8
- **🎯 Seguimiento con DeepSORT**: Seguimiento robusto con IDs consistentes
- **📏 Cálculo de Métricas**: Velocidad, distancia recorrida y análisis de trayectorias
- **🎨 Visualización en Tiempo Real**: Renderizado avanzado con trayectorias y estadísticas
- **🎬 Grabación y GIFs**: Captura automática de escenas interesantes
- **🖥️ Interfaz Gráfica**: GUI intuitiva con PySimpleGUI
- **⚙️ Configuración Flexible**: Sistema de configuración YAML completo

## 🏗️ Arquitectura del Sistema

```
[Video Input] → FrameReader → Detector (YOLO) → Tracker (DeepSORT) 
                                    ↓
[Output & GIFs] ← Recorder ← Visualizer ← MetricsCalculator
```

### Componentes Principales

| Componente | Función | Tecnología |
|------------|---------|------------|
| **FrameReader** | Lectura de video/cámara | OpenCV |
| **Detector** | Detección de objetos | YOLOv8 (Ultralytics) |
| **Tracker** | Seguimiento de objetos | DeepSORT |
| **MetricsCalculator** | Cálculo de velocidad/distancia | NumPy/SciPy |
| **Visualizer** | Renderizado en tiempo real | OpenCV |
| **Recorder** | Grabación de videos/GIFs | ImageIO |

## 🚀 Instalación

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd parcial-visual
```

### 2. Crear Entorno Virtual (Recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar Modelos (Opcional)

Los modelos YOLO se descargan automáticamente en el primer uso. Para modelos específicos:

```bash
# Crear carpeta de modelos
mkdir models

# Descargar modelos manualmente (opcional)
# Los modelos se descargan automáticamente: yolov8n.pt, yolov8s.pt, etc.
```

## 🎮 Uso del Sistema

### Modo Comando (Terminal)

```bash
# Usar cámara por defecto
python main.py

# Usar archivo de video específico
python main.py --source video.mp4

# Usar cámara específica
python main.py --source 1

# Modo headless (sin ventana)
python main.py --headless --record

# Configuración personalizada
python main.py --confidence 0.7 --device cuda --pixels-per-meter 100
```

#### Controles del Teclado

- **Q**: Salir del programa
- **ESPACIO**: Pausar/Reanudar procesamiento
- **R**: Iniciar/Detener grabación manual
- **G**: Crear GIF del buffer actual
- **S**: Mostrar estadísticas detalladas

### Modo Interfaz Gráfica

```bash
python gui_app.py
```

La interfaz gráfica proporciona:
- Control visual de todos los parámetros
- Visualización en tiempo real
- Estadísticas actualizadas
- Controles de grabación intuitivos

## ⚙️ Configuración

El sistema utiliza un archivo de configuración YAML ubicado en `config/config.yaml`:

### Configuración del Detector YOLO

```yaml
detector:
  model_path: "models/yolov8n.pt"  # Modelo a usar
  confidence_threshold: 0.5         # Umbral de confianza
  iou_threshold: 0.45              # Umbral IoU para NMS
  device: "cpu"                    # "cpu" o "cuda"
  classes_to_detect: []            # Clases específicas o [] para todas
```

### Configuración del Tracker

```yaml
tracker:
  max_age: 50          # Frames máximos sin detección
  min_hits: 3          # Detecciones mínimas para confirmar
  iou_threshold: 0.3   # Umbral IoU para asociación
```

### Configuración de Métricas

```yaml
metrics:
  pixels_per_meter: 50  # Calibración espacial
  fps: 30              # FPS del video
  velocity_smoothing_window: 10  # Ventana para suavizado
```

### Configuración del Visualizador

```yaml
visualizer:
  show_trajectory: true      # Mostrar trayectorias
  trajectory_length: 30      # Puntos en trayectoria
  show_velocity: true        # Mostrar velocidad
  show_distance: true        # Mostrar distancia
```

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

## 🎬 Salida del Sistema

### Videos Generados

- `session_YYYYMMDD_HHMMSS.mp4`: Grabaciones de sesión completa
- `auto_recording_YYYYMMDD_HHMMSS.mp4`: Grabaciones automáticas de escenas activas
- `manual_HHMMSS.mp4`: Grabaciones manuales iniciadas por el usuario

### GIFs Creados

- `tracking_gif_YYYYMMDD_HHMMSS.gif`: GIFs del buffer actual
- `session_final.gif`: GIF resumen de la sesión
- `auto_recording_YYYYMMDD_HHMMSS.gif`: GIFs automáticos de escenas activas

### Métricas Exportables

```python
# Exportar métricas a CSV
metrics_calculator.export_metrics_csv('output/metrics.csv')
```

## 📊 Métricas Calculadas

### Por Objeto

- **Velocidad instantánea**: En píxeles/frame y m/s
- **Velocidad promedio**: Suavizada en ventana temporal
- **Distancia recorrida**: Total acumulada en metros
- **Aceleración**: Cambio de velocidad en m/s²
- **Suavidad de trayectoria**: Índice de consistencia (0-1)

### Globales del Sistema

- **FPS de procesamiento**: Rendimiento en tiempo real
- **Número de tracks activos**: Objetos siendo seguidos
- **Estadísticas de velocidad**: Promedio y máximo
- **Distancia total**: Suma de todas las trayectorias

## 🔧 Personalización y Extensión

### Agregar Nuevas Clases de Detección

```python
# En config.yaml
detector:
  classes_to_detect: [0, 1, 2]  # person, bicycle, car
```

### Calibración Espacial

Para medir distancias reales:

1. Medir una distancia conocida en píxeles en el video
2. Calcular `pixels_per_meter = pixels_medidos / metros_reales`
3. Actualizar en configuración o GUI

### Personalizar Visualización

```python
# Modificar colores en config.yaml
visualizer:
  bbox_colors:
    - [255, 0, 0]    # Azul
    - [0, 255, 0]    # Verde
    - [0, 0, 255]    # Rojo
```

## 🐛 Solución de Problemas

### Error: "YOLO no está disponible"

```bash
pip install ultralytics
```

### Error: "DeepSORT no está disponible"

```bash
pip install deep-sort-realtime
```

### Error: "No se puede abrir la cámara"

- Verificar que la cámara no esté en uso por otra aplicación
- Probar con diferente `camera_id` (0, 1, 2...)
- En Linux, verificar permisos de cámara

### Error: "Modelo YOLO no encontrado"

- Los modelos se descargan automáticamente en el primer uso
- Verificar conexión a internet
- Usar modelo local: colocar archivo `.pt` en carpeta `models/`

### Rendimiento Lento

1. **Usar modelo YOLO más ligero**: `yolov8n.pt` en lugar de `yolov8x.pt`
2. **Reducir resolución**: Configurar `frame_width` y `frame_height` más bajos
3. **Usar GPU**: Instalar PyTorch con CUDA y configurar `device: "cuda"`
4. **Saltar frames**: Configurar `skip_frames > 0`

## 🏆 Casos de Uso

### Análisis de Tráfico

```yaml
# Configuración para vehículos
detector:
  classes_to_detect: [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
metrics:
  pixels_per_meter: 20  # Ajustar según distancia de cámara
```

### Análisis Deportivo

```yaml
# Configuración para personas
detector:
  classes_to_detect: [0]  # person
  confidence_threshold: 0.3  # Más sensible
metrics:
  pixels_per_meter: 100  # Campo más cercano
  velocity_smoothing_window: 5  # Respuesta más rápida
```

### Seguridad y Vigilancia

```yaml
# Detección general con auto-grabación
detector:
  confidence_threshold: 0.6  # Menos falsos positivos
recorder:
  auto_record_interesting_scenes: true
  min_objects_for_recording: 1  # Grabar con cualquier objeto
```

## 🤝 Contribución

1. Fork el repositorio
2. Crear rama para nueva característica: `git checkout -b feature/nueva-caracteristica`
3. Commit cambios: `git commit -m 'Agregar nueva característica'`
4. Push a la rama: `git push origin feature/nueva-caracteristica`
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **Ultralytics** por YOLOv8
- **DeepSORT** por el algoritmo de tracking
- **OpenCV** por las herramientas de visión por computador
- **PySimpleGUI** por la interfaz gráfica

## 📞 Soporte

Para problemas, preguntas o sugerencias:

1. Revisar la sección de [Solución de Problemas](#-solución-de-problemas)
2. Buscar en [Issues existentes](../../issues)
3. Crear un [nuevo Issue](../../issues/new) si es necesario

---

**Desarrollado para el Parcial de Visual Computing** 🎓 