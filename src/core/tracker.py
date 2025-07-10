"""
Tracker de Objetos - DeepSORT
=============================

Este módulo implementa el seguimiento de objetos usando DeepSORT
para mantener IDs consistentes y trayectorias a través del tiempo.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
from collections import deque

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("⚠️ DeepSORT no está instalado. Ejecute: pip install deep-sort-realtime")

from .detector import Detection
from ..utils.config_loader import get_config


@dataclass
class Track:
    """
    Estructura de datos para un track (objeto seguido).
    
    Attributes:
        track_id: ID único del track
        bbox: Bounding box actual en formato [x1, y1, x2, y2]
        confidence: Confianza de la detección
        class_id: ID de la clase del objeto
        class_name: Nombre de la clase del objeto
        age: Edad del track (frames desde creación)
        hits: Número de veces que ha sido detectado
        time_since_update: Frames desde la última actualización
        trajectory: Historial de posiciones del centro
        velocities: Historial de velocidades
        is_confirmed: Si el track está confirmado
    """
    track_id: int
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    velocities: deque = field(default_factory=lambda: deque(maxlen=10))
    is_confirmed: bool = False
    
    def __post_init__(self):
        """Inicialización post-creación."""
        # Agregar posición inicial a la trayectoria
        center = self.center
        self.trajectory.append((center[0], center[1], time.time()))
    
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
    
    def update_position(self, bbox: List[float], confidence: float) -> None:
        """
        Actualiza la posición del track.
        
        Args:
            bbox: Nuevo bounding box
            confidence: Nueva confianza
        """
        # Calcular velocidad si hay trayectoria previa
        if len(self.trajectory) > 0:
            prev_center = self.trajectory[-1][:2]
            current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Calcular velocidad en píxeles por frame
            velocity = (
                current_center[0] - prev_center[0],
                current_center[1] - prev_center[1]
            )
            self.velocities.append(velocity)
        
        # Actualizar bounding box y confianza
        self.bbox = bbox
        self.confidence = confidence
        
        # Agregar nueva posición a la trayectoria
        center = self.center
        self.trajectory.append((center[0], center[1], time.time()))
        
        # Actualizar estadísticas
        self.hits += 1
        self.time_since_update = 0
    
    def predict_next_position(self) -> Tuple[float, float]:
        """
        Predice la siguiente posición basada en la velocidad.
        
        Returns:
            Posición predicha (x, y)
        """
        if len(self.velocities) == 0:
            return self.center
        
        # Usar velocidad promedio de las últimas mediciones
        avg_velocity = np.mean(list(self.velocities), axis=0)
        current_center = self.center
        
        predicted_center = (
            current_center[0] + avg_velocity[0],
            current_center[1] + avg_velocity[1]
        )
        
        return predicted_center
    
    def get_average_velocity(self) -> Tuple[float, float]:
        """
        Obtiene la velocidad promedio del track.
        
        Returns:
            Velocidad promedio (vx, vy) en píxeles por frame
        """
        if len(self.velocities) == 0:
            return (0.0, 0.0)
        
        return tuple(np.mean(list(self.velocities), axis=0))
    
    def get_speed(self) -> float:
        """
        Obtiene la velocidad escalar del track.
        
        Returns:
            Velocidad en píxeles por frame
        """
        vx, vy = self.get_average_velocity()
        return np.sqrt(vx**2 + vy**2)


class ObjectTracker:
    """
    Tracker de objetos usando DeepSORT.
    
    Mantiene la identidad de los objetos a través del tiempo
    y proporciona información de trayectorias.
    """
    
    def __init__(self):
        """Inicializa el tracker."""
        if not DEEPSORT_AVAILABLE:
            print("⚠️ DeepSORT no disponible. Usando tracker simple...")
            self.use_deepsort = False
            self.simple_tracker = SimpleTracker()
        else:
            self.use_deepsort = True
            self._load_config()
            self._initialize_deepsort()
        
        # Estadísticas
        self.total_tracks = 0
        self.active_tracks = 0
        self.tracking_times = []
    
    def _load_config(self) -> None:
        """Carga la configuración del tracker."""
        self.max_age = get_config('tracker.max_age', 50)
        self.min_hits = get_config('tracker.min_hits', 3)
        self.iou_threshold = get_config('tracker.iou_threshold', 0.3)
        self.max_iou_distance = get_config('tracker.max_iou_distance', 0.7)
        self.max_cosine_distance = get_config('tracker.max_cosine_distance', 0.2)
        self.nn_budget = get_config('tracker.nn_budget', 100)
        
        print(f"🎯 Configuración del tracker:")
        print(f"   ⏰ Max age: {self.max_age}")
        print(f"   🎯 Min hits: {self.min_hits}")
        print(f"   📏 IoU threshold: {self.iou_threshold}")
    
    def _initialize_deepsort(self) -> None:
        """Inicializa DeepSORT."""
        try:
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.min_hits,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget
            )
            print("✅ DeepSORT inicializado exitosamente")
        except Exception as e:
            print(f"❌ Error inicializando DeepSORT: {e}")
            print("🔄 Cambiando a tracker simple...")
            self.use_deepsort = False
            self.simple_tracker = SimpleTracker()
    
    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Actualiza el tracker con nuevas detecciones.
        
        Args:
            detections: Lista de detecciones del frame actual
            frame: Frame actual (opcional, para DeepSORT)
            
        Returns:
            Lista de tracks actualizados
        """
        start_time = time.time()
        
        if self.use_deepsort:
            tracks = self._update_deepsort(detections, frame)
        else:
            tracks = self._update_simple(detections)
        
        # Estadísticas
        tracking_time = time.time() - start_time
        self.tracking_times.append(tracking_time)
        self.active_tracks = len(tracks)
        
        # Mantener solo los últimos 100 tiempos
        if len(self.tracking_times) > 100:
            self.tracking_times = self.tracking_times[-100:]
        
        return tracks
    
    def _update_deepsort(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Actualiza usando DeepSORT.
        
        Args:
            detections: Lista de detecciones
            frame: Frame actual
            
        Returns:
            Lista de tracks
        """
        if not detections:
            # Actualizar tracker sin detecciones
            tracks = self.tracker.update_tracks([], frame=frame)
            return self._convert_deepsort_tracks(tracks)
        
        # Convertir detecciones al formato de DeepSORT
        raw_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            raw_detections.append(([x1, y1, x2 - x1, y2 - y1], det.confidence, det.class_name))
        
        # Actualizar tracker
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        return self._convert_deepsort_tracks(tracks, detections)
    
    def _convert_deepsort_tracks(self, deepsort_tracks: List, detections: List[Detection] = None) -> List[Track]:
        """
        Convierte tracks de DeepSORT al formato interno.
        
        Args:
            deepsort_tracks: Tracks de DeepSORT
            detections: Detecciones originales para obtener información de clase
            
        Returns:
            Lista de tracks convertidos
        """
        tracks = []
        
        for track in deepsort_tracks:
            if not track.is_confirmed():
                continue
            
            # Obtener bounding box
            ltrb = track.to_ltrb()
            bbox = [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
            
            # Intentar obtener información de clase de la detección más cercana
            class_id = 0
            class_name = "unknown"
            confidence = 0.5
            
            if detections:
                # Encontrar la detección más cercana
                track_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                min_dist = float('inf')
                best_det = None
                
                for det in detections:
                    det_center = det.center
                    dist = np.sqrt((track_center[0] - det_center[0])**2 + 
                                 (track_center[1] - det_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_det = det
                
                if best_det:
                    class_id = best_det.class_id
                    class_name = best_det.class_name
                    confidence = best_det.confidence
            
            # Crear track
            track_obj = Track(
                track_id=track.track_id,
                bbox=bbox,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
                age=track.age,
                hits=track.hits,
                time_since_update=track.time_since_update,
                is_confirmed=track.is_confirmed()
            )
            
            tracks.append(track_obj)
        
        return tracks
    
    def _update_simple(self, detections: List[Detection]) -> List[Track]:
        """
        Actualiza usando tracker simple.
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Lista de tracks
        """
        return self.simple_tracker.update(detections)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del tracker.
        
        Returns:
            Diccionario con estadísticas
        """
        avg_tracking_time = np.mean(self.tracking_times) if self.tracking_times else 0
        
        # Obtener configuración según el tipo de tracker
        if self.use_deepsort:
            max_age = getattr(self, 'max_age', 50)
            min_hits = getattr(self, 'min_hits', 3)
        else:
            max_age = getattr(self.simple_tracker, 'max_age', 30)
            min_hits = getattr(self.simple_tracker, 'min_hits', 3)
        
        return {
            'tracker_type': 'DeepSORT' if self.use_deepsort else 'Simple',
            'total_tracks': self.total_tracks,
            'active_tracks': self.active_tracks,
            'avg_tracking_time': avg_tracking_time,
            'max_age': max_age,
            'min_hits': min_hits
        }


class SimpleTracker:
    """
    Tracker simple basado en IoU para casos donde DeepSORT no esté disponible.
    """
    
    def __init__(self):
        """Inicializa el tracker simple."""
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.max_age = 30
        self.min_hits = 3
        self.iou_threshold = 0.3
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Actualiza el tracker con nuevas detecciones.
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Lista de tracks activos
        """
        # Incrementar edad de todos los tracks
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1
        
        # Asociar detecciones con tracks existentes
        if detections:
            self._associate_detections(detections)
        
        # Eliminar tracks antiguos
        self._remove_old_tracks()
        
        # Retornar tracks confirmados
        return [track for track in self.tracks.values() if track.is_confirmed]
    
    def _associate_detections(self, detections: List[Detection]) -> None:
        """
        Asocia detecciones con tracks existentes.
        
        Args:
            detections: Lista de detecciones
        """
        # Calcular matriz de IoU
        track_list = list(self.tracks.values())
        iou_matrix = np.zeros((len(track_list), len(detections)))
        
        for i, track in enumerate(track_list):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection.bbox)
        
        # Asociación simple: mejor IoU por encima del umbral
        used_detections = set()
        used_tracks = set()
        
        # Ordenar por IoU descendente
        matches = []
        for i in range(len(track_list)):
            for j in range(len(detections)):
                if j not in used_detections and i not in used_tracks:
                    if iou_matrix[i, j] > self.iou_threshold:
                        matches.append((i, j, iou_matrix[i, j]))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Actualizar tracks asociados
        for track_idx, det_idx, _ in matches:
            if track_idx not in used_tracks and det_idx not in used_detections:
                track = track_list[track_idx]
                detection = detections[det_idx]
                
                track.update_position(detection.bbox, detection.confidence)
                track.class_id = detection.class_id
                track.class_name = detection.class_name
                
                if track.hits >= self.min_hits:
                    track.is_confirmed = True
                
                used_tracks.add(track_idx)
                used_detections.add(det_idx)
        
        # Crear nuevos tracks para detecciones no asociadas
        for j, detection in enumerate(detections):
            if j not in used_detections:
                new_track = Track(
                    track_id=self.next_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    hits=1
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calcula el IoU entre dos bounding boxes.
        
        Args:
            bbox1: Primer bounding box [x1, y1, x2, y2]
            bbox2: Segundo bounding box [x1, y1, x2, y2]
            
        Returns:
            Valor de IoU (0-1)
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_old_tracks(self) -> None:
        """Elimina tracks que han estado demasiado tiempo sin actualizar."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id] 