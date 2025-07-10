"""
Detector de Objetos - YOLOv8
============================

Este módulo implementa la detección de objetos usando YOLOv8
de Ultralytics con configuración flexible y optimización.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO no está instalado. Ejecute: pip install ultralytics")

from ..utils.config_loader import get_config


@dataclass
class Detection:
    """
    Estructura de datos para una detección.
    
    Attributes:
        bbox: Bounding box en formato [x1, y1, x2, y2]
        confidence: Confianza de la detección (0-1)
        class_id: ID de la clase detectada
        class_name: Nombre de la clase detectada
    """
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """Obtiene el centro del bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """Obtiene el ancho del bounding box."""
        x1, _, x2, _ = self.bbox
        return x2 - x1
    
    @property
    def height(self) -> float:
        """Obtiene la altura del bounding box."""
        _, y1, _, y2 = self.bbox
        return y2 - y1
    
    @property
    def area(self) -> float:
        """Obtiene el área del bounding box."""
        return self.width * self.height


class YOLODetector:
    """
    Detector de objetos usando YOLOv8.
    
    Proporciona detección de objetos con configuración flexible
    y optimización para tiempo real.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el detector YOLO.
        
        Args:
            model_path: Ruta al modelo YOLO. Si es None, usa configuración
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO no está disponible. Instale ultralytics: pip install ultralytics")
        
        self.model = None
        self.model_path = model_path
        self.class_names = []
        
        # Cargar configuración
        self._load_config()
        
        # Estadísticas
        self.detection_times = []
        self.total_detections = 0
        
        # Inicializar modelo
        self._initialize_model()
    
    def _load_config(self) -> None:
        """Carga la configuración del detector."""
        if self.model_path is None:
            self.model_path = get_config('detector.model_path', 'yolov8n.pt')
        
        self.confidence_threshold = get_config('detector.confidence_threshold', 0.5)
        self.iou_threshold = get_config('detector.iou_threshold', 0.45)
        self.device = get_config('detector.device', 'cpu')
        self.classes_to_detect = get_config('detector.classes_to_detect', [])
        self.input_size = get_config('detector.input_size', [640, 640])
        
        print(f"🎯 Configuración del detector:")
        print(f"   📄 Modelo: {self.model_path}")
        print(f"   🎯 Confianza: {self.confidence_threshold}")
        print(f"   💻 Dispositivo: {self.device}")
    
    def _initialize_model(self) -> None:
        """Inicializa el modelo YOLO."""
        try:
            print(f"🔄 Cargando modelo YOLO: {self.model_path}")
            
            # Verificar si el archivo existe
            model_file = Path(self.model_path)
            if not model_file.exists() and not self.model_path.endswith('.pt'):
                # Intentar descargar modelo predefinido
                print(f"📥 Descargando modelo {self.model_path}...")
            
            self.model = YOLO(self.model_path)
            
            # Obtener nombres de clases
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                # Nombres COCO por defecto
                self.class_names = self._get_coco_class_names()
            
            print(f"✅ Modelo YOLO cargado exitosamente")
            print(f"   📊 Clases disponibles: {len(self.class_names)}")
            
            # Calentar el modelo con una imagen dummy
            self._warmup_model()
            
        except Exception as e:
            print(f"❌ Error cargando modelo YOLO: {e}")
            raise
    
    def _warmup_model(self) -> None:
        """Calienta el modelo con una inferencia dummy."""
        try:
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            start_time = time.time()
            _ = self.model(dummy_image, verbose=False)
            warmup_time = time.time() - start_time
            print(f"🔥 Modelo calentado en {warmup_time:.3f}s")
        except Exception as e:
            print(f"⚠️ Error en warmup: {e}")
    
    def _get_coco_class_names(self) -> List[str]:
        """Obtiene los nombres de clases COCO por defecto."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detecta objetos en un frame.
        
        Args:
            frame: Frame de entrada como array numpy
            
        Returns:
            Lista de detecciones encontradas
        """
        if self.model is None:
            return []
        
        start_time = time.time()
        
        try:
            # Ejecutar inferencia
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=self.input_size
            )
            
            detections = []
            
            # Procesar resultados
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Filtrar por clases específicas si está configurado
                        if self.classes_to_detect and class_id not in self.classes_to_detect:
                            continue
                        
                        # Obtener nombre de clase
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detection = Detection(
                            bbox=box.tolist(),
                            confidence=float(conf),
                            class_id=int(class_id),
                            class_name=class_name
                        )
                        
                        detections.append(detection)
            
            # Estadísticas
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.total_detections += len(detections)
            
            # Mantener solo los últimos 100 tiempos para estadísticas
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]
            
            return detections
            
        except Exception as e:
            print(f"❌ Error en detección: {e}")
            return []
    
    def detect_and_visualize(self, frame: np.ndarray, draw_boxes: bool = True) -> Tuple[List[Detection], np.ndarray]:
        """
        Detecta objetos y los visualiza en el frame.
        
        Args:
            frame: Frame de entrada
            draw_boxes: Si dibujar los bounding boxes
            
        Returns:
            Tupla (detecciones, frame_con_visualización)
        """
        detections = self.detect(frame)
        
        if draw_boxes:
            frame = self._draw_detections(frame, detections)
        
        return detections, frame
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Dibuja las detecciones en el frame.
        
        Args:
            frame: Frame original
            detections: Lista de detecciones
            
        Returns:
            Frame con detecciones dibujadas
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Color basado en la clase
            color = self._get_class_color(detection.class_id)
            
            # Dibujar bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Preparar texto
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Obtener tamaño del texto
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Dibujar fondo del texto
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return frame_copy
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Obtiene un color consistente para una clase.
        
        Args:
            class_id: ID de la clase
            
        Returns:
            Color en formato BGR
        """
        # Generar colores consistentes basados en el ID de clase
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 256, 3)))
        return color
    
    def get_supported_classes(self) -> List[str]:
        """
        Obtiene la lista de clases soportadas.
        
        Returns:
            Lista de nombres de clases
        """
        return self.class_names.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del detector.
        
        Returns:
            Diccionario con estadísticas
        """
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
        
        return {
            'total_detections': self.total_detections,
            'avg_detection_time': avg_detection_time,
            'fps': fps,
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'supported_classes': len(self.class_names)
        }
    
    def update_threshold(self, confidence: float) -> None:
        """
        Actualiza el umbral de confianza.
        
        Args:
            confidence: Nuevo umbral de confianza (0-1)
        """
        self.confidence_threshold = max(0.0, min(1.0, confidence))
        print(f"🎯 Umbral de confianza actualizado: {self.confidence_threshold}")
    
    def set_classes_filter(self, class_names: List[str]) -> None:
        """
        Configura filtro de clases específicas.
        
        Args:
            class_names: Lista de nombres de clases a detectar
        """
        class_ids = []
        for name in class_names:
            try:
                class_id = self.class_names.index(name)
                class_ids.append(class_id)
            except ValueError:
                print(f"⚠️ Clase no encontrada: {name}")
        
        self.classes_to_detect = class_ids
        print(f"🎯 Filtro de clases configurado: {class_names}") 