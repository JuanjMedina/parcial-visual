"""
Calculadora de Métricas - MetricsCalculator
=========================================

Este módulo calcula métricas físicas (velocidad, distancia recorrida)
a partir de las trayectorias de los objetos tracked.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
from collections import deque

from .tracker import Track
from ..utils.config_loader import get_config


@dataclass
class ObjectMetrics:
    """
    Métricas calculadas para un objeto.
    
    Attributes:
        track_id: ID del track
        speed_pixels_per_frame: Velocidad en píxeles por frame
        speed_pixels_per_second: Velocidad en píxeles por segundo
        speed_meters_per_second: Velocidad en metros por segundo
        distance_traveled_pixels: Distancia total recorrida en píxeles
        distance_traveled_meters: Distancia total recorrida en metros
        average_speed_mps: Velocidad promedio en m/s
        current_acceleration: Aceleración actual en m/s²
        trajectory_smoothness: Suavidad de la trayectoria (0-1)
    """
    track_id: int
    speed_pixels_per_frame: float = 0.0
    speed_pixels_per_second: float = 0.0
    speed_meters_per_second: float = 0.0
    distance_traveled_pixels: float = 0.0
    distance_traveled_meters: float = 0.0
    average_speed_mps: float = 0.0
    current_acceleration: float = 0.0
    trajectory_smoothness: float = 1.0


class MetricsCalculator:
    """
    Calculadora de métricas físicas para objetos tracked.
    
    Convierte mediciones en píxeles a unidades del mundo real
    y calcula velocidades, distancias y otras métricas.
    """
    
    def __init__(self):
        """Inicializa la calculadora de métricas."""
        # Cargar configuración
        self._load_config()
        
        # Historial de métricas por track
        self.metrics_history: Dict[int, List[ObjectMetrics]] = {}
        
        # Cache de cálculos previos
        self.previous_positions: Dict[int, Tuple[float, float, float]] = {}  # (x, y, timestamp)
        self.previous_speeds: Dict[int, deque] = {}
        
        # Estadísticas generales
        self.total_calculations = 0
        self.calculation_times = []
    
    def _load_config(self) -> None:
        """Carga la configuración de métricas."""
        self.pixels_per_meter = get_config('metrics.pixels_per_meter', 50)
        self.fps = get_config('metrics.fps', 30)
        self.velocity_smoothing_window = get_config('metrics.velocity_smoothing_window', 10)
        self.min_movement_threshold = get_config('metrics.min_movement_threshold', 5)
        
        print(f"📏 Configuración de métricas:")
        print(f"   🎯 Píxeles por metro: {self.pixels_per_meter}")
        print(f"   📹 FPS: {self.fps}")
        print(f"   📊 Ventana de suavizado: {self.velocity_smoothing_window}")
    
    def update_calibration(self, pixels_per_meter: float, fps: float) -> None:
        """
        Actualiza la calibración del sistema.
        
        Args:
            pixels_per_meter: Cuántos píxeles equivalen a un metro
            fps: Frames por segundo del video
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        print(f"🔄 Calibración actualizada: {pixels_per_meter} px/m, {fps} FPS")
    
    def calculate_metrics(self, tracks: List[Track]) -> Dict[int, ObjectMetrics]:
        """
        Calcula métricas para todos los tracks.
        
        Args:
            tracks: Lista de tracks activos
            
        Returns:
            Diccionario con métricas por track_id
        """
        start_time = time.time()
        metrics_dict = {}
        
        for track in tracks:
            if not track.is_confirmed or len(track.trajectory) < 2:
                continue
            
            metrics = self._calculate_track_metrics(track)
            metrics_dict[track.track_id] = metrics
            
            # Guardar en historial
            if track.track_id not in self.metrics_history:
                self.metrics_history[track.track_id] = []
            self.metrics_history[track.track_id].append(metrics)
            
            # Mantener solo las últimas 100 métricas por track
            if len(self.metrics_history[track.track_id]) > 100:
                self.metrics_history[track.track_id] = self.metrics_history[track.track_id][-100:]
        
        # Estadísticas
        calculation_time = time.time() - start_time
        self.calculation_times.append(calculation_time)
        self.total_calculations += 1
        
        # Mantener solo los últimos 100 tiempos
        if len(self.calculation_times) > 100:
            self.calculation_times = self.calculation_times[-100:]
        
        return metrics_dict
    
    def _calculate_track_metrics(self, track: Track) -> ObjectMetrics:
        """
        Calcula métricas para un track específico.
        
        Args:
            track: Track a analizar
            
        Returns:
            Métricas calculadas
        """
        metrics = ObjectMetrics(track_id=track.track_id)
        
        # Obtener posición actual
        current_pos = track.center
        current_time = time.time()
        
        # Calcular velocidad instantánea
        if track.track_id in self.previous_positions:
            prev_x, prev_y, prev_time = self.previous_positions[track.track_id]
            
            # Calcular distancia en píxeles
            distance_pixels = np.sqrt((current_pos[0] - prev_x)**2 + (current_pos[1] - prev_y)**2)
            time_diff = current_time - prev_time
            
            if time_diff > 0 and distance_pixels > self.min_movement_threshold:
                # Velocidad en píxeles por segundo
                speed_pps = distance_pixels / time_diff
                metrics.speed_pixels_per_second = speed_pps
                
                # Velocidad en píxeles por frame
                metrics.speed_pixels_per_frame = speed_pps / self.fps
                
                # Velocidad en metros por segundo
                metrics.speed_meters_per_second = speed_pps / self.pixels_per_meter
                
                # Almacenar velocidad para promedio
                if track.track_id not in self.previous_speeds:
                    self.previous_speeds[track.track_id] = deque(maxlen=self.velocity_smoothing_window)
                self.previous_speeds[track.track_id].append(metrics.speed_meters_per_second)
        
        # Actualizar posición previa
        self.previous_positions[track.track_id] = (current_pos[0], current_pos[1], current_time)
        
        # Calcular velocidad promedio
        if track.track_id in self.previous_speeds and len(self.previous_speeds[track.track_id]) > 0:
            metrics.average_speed_mps = np.mean(list(self.previous_speeds[track.track_id]))
        
        # Calcular distancia total recorrida
        metrics.distance_traveled_pixels = self._calculate_total_distance_pixels(track)
        metrics.distance_traveled_meters = metrics.distance_traveled_pixels / self.pixels_per_meter
        
        # Calcular aceleración
        metrics.current_acceleration = self._calculate_acceleration(track)
        
        # Calcular suavidad de la trayectoria
        metrics.trajectory_smoothness = self._calculate_trajectory_smoothness(track)
        
        return metrics
    
    def _calculate_total_distance_pixels(self, track: Track) -> float:
        """
        Calcula la distancia total recorrida por un track en píxeles.
        
        Args:
            track: Track a analizar
            
        Returns:
            Distancia total en píxeles
        """
        if len(track.trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        trajectory_list = list(track.trajectory)
        
        for i in range(1, len(trajectory_list)):
            prev_point = trajectory_list[i-1]
            current_point = trajectory_list[i]
            
            # Calcular distancia entre puntos consecutivos
            distance = np.sqrt(
                (current_point[0] - prev_point[0])**2 + 
                (current_point[1] - prev_point[1])**2
            )
            total_distance += distance
        
        return total_distance
    
    def _calculate_acceleration(self, track: Track) -> float:
        """
        Calcula la aceleración actual del track.
        
        Args:
            track: Track a analizar
            
        Returns:
            Aceleración en m/s²
        """
        if track.track_id not in self.previous_speeds:
            return 0.0
        
        speeds = list(self.previous_speeds[track.track_id])
        if len(speeds) < 2:
            return 0.0
        
        # Calcular aceleración como cambio de velocidad
        recent_speeds = speeds[-3:]  # Últimas 3 velocidades
        if len(recent_speeds) >= 2:
            time_per_measurement = 1.0 / self.fps  # Tiempo entre mediciones
            acceleration = (recent_speeds[-1] - recent_speeds[0]) / (len(recent_speeds) * time_per_measurement)
            return acceleration
        
        return 0.0
    
    def _calculate_trajectory_smoothness(self, track: Track) -> float:
        """
        Calcula la suavidad de la trayectoria (0 = muy errática, 1 = muy suave).
        
        Args:
            track: Track a analizar
            
        Returns:
            Índice de suavidad (0-1)
        """
        if len(track.trajectory) < 3:
            return 1.0
        
        trajectory_list = list(track.trajectory)
        
        # Calcular cambios de dirección
        direction_changes = []
        
        for i in range(2, len(trajectory_list)):
            p1 = trajectory_list[i-2]
            p2 = trajectory_list[i-1]
            p3 = trajectory_list[i]
            
            # Vectores de dirección
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            # Normalizar vectores
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                v1_norm = v1 / norm1
                v2_norm = v2 / norm2
                
                # Producto punto para obtener el ángulo
                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle_change = np.arccos(dot_product)
                direction_changes.append(angle_change)
        
        if not direction_changes:
            return 1.0
        
        # Calcular suavidad como inverso de la varianza de cambios de dirección
        variance = np.var(direction_changes)
        smoothness = 1.0 / (1.0 + variance)
        
        return smoothness
    
    def get_track_history(self, track_id: int, last_n: Optional[int] = None) -> List[ObjectMetrics]:
        """
        Obtiene el historial de métricas de un track.
        
        Args:
            track_id: ID del track
            last_n: Número de métricas más recientes (None = todas)
            
        Returns:
            Lista de métricas históricas
        """
        if track_id not in self.metrics_history:
            return []
        
        history = self.metrics_history[track_id]
        if last_n is not None:
            return history[-last_n:]
        return history.copy()
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas globales del sistema.
        
        Returns:
            Diccionario con estadísticas
        """
        all_speeds = []
        all_distances = []
        active_tracks = len(self.metrics_history)
        
        # Recopilar todas las velocidades y distancias
        for track_metrics in self.metrics_history.values():
            if track_metrics:
                latest = track_metrics[-1]
                all_speeds.append(latest.speed_meters_per_second)
                all_distances.append(latest.distance_traveled_meters)
        
        avg_calculation_time = np.mean(self.calculation_times) if self.calculation_times else 0
        
        return {
            'total_calculations': self.total_calculations,
            'active_tracks': active_tracks,
            'avg_calculation_time': avg_calculation_time,
            'avg_speed_mps': np.mean(all_speeds) if all_speeds else 0,
            'max_speed_mps': np.max(all_speeds) if all_speeds else 0,
            'avg_distance_m': np.mean(all_distances) if all_distances else 0,
            'total_distance_m': np.sum(all_distances) if all_distances else 0,
            'pixels_per_meter': self.pixels_per_meter,
            'fps': self.fps
        }
    
    def export_metrics_csv(self, filepath: str) -> None:
        """
        Exporta las métricas a un archivo CSV.
        
        Args:
            filepath: Ruta del archivo CSV a crear
        """
        import csv
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'track_id', 'timestamp', 'speed_mps', 'distance_m',
                    'acceleration_mps2', 'trajectory_smoothness'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for track_id, metrics_list in self.metrics_history.items():
                    for i, metrics in enumerate(metrics_list):
                        writer.writerow({
                            'track_id': track_id,
                            'timestamp': i,
                            'speed_mps': metrics.speed_meters_per_second,
                            'distance_m': metrics.distance_traveled_meters,
                            'acceleration_mps2': metrics.current_acceleration,
                            'trajectory_smoothness': metrics.trajectory_smoothness
                        })
            
            print(f"✅ Métricas exportadas a: {filepath}")
            
        except Exception as e:
            print(f"❌ Error exportando métricas: {e}")
    
    def clear_track_history(self, track_id: Optional[int] = None) -> None:
        """
        Limpia el historial de métricas.
        
        Args:
            track_id: ID del track a limpiar (None = limpiar todo)
        """
        if track_id is None:
            self.metrics_history.clear()
            self.previous_positions.clear()
            self.previous_speeds.clear()
            print("🧹 Historial de métricas limpiado completamente")
        else:
            if track_id in self.metrics_history:
                del self.metrics_history[track_id]
            if track_id in self.previous_positions:
                del self.previous_positions[track_id]
            if track_id in self.previous_speeds:
                del self.previous_speeds[track_id]
            print(f"🧹 Historial del track {track_id} limpiado")
    
    def pixels_to_meters(self, pixels: float) -> float:
        """
        Convierte píxeles a metros.
        
        Args:
            pixels: Distancia en píxeles
            
        Returns:
            Distancia en metros
        """
        return pixels / self.pixels_per_meter
    
    def meters_to_pixels(self, meters: float) -> float:
        """
        Convierte metros a píxeles.
        
        Args:
            meters: Distancia en metros
            
        Returns:
            Distancia en píxeles
        """
        return meters * self.pixels_per_meter 