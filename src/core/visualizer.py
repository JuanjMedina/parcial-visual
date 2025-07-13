"""
Visualizador - Visualizer
=========================

Este módulo maneja la visualización en tiempo real de detecciones,
tracks, trayectorias y métricas sobre los frames de video.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time
from collections import deque

from .tracker import Track
from .detector import Detection
from .metrics_calculator import ObjectMetrics
from ..utils.config_loader import get_config


class Visualizer:
    """
    Visualizador en tiempo real para el sistema de tracking.
    
    Renderiza detecciones, tracks, trayectorias y métricas
    de forma clara y atractiva.
    """
    
    def __init__(self):
        """Inicializa el visualizador."""
        # Cargar configuración
        self._load_config()
        
        # Cache de colores por track
        self.track_colors: Dict[int, Tuple[int, int, int]] = {}
        
        # Trayectorias para visualización
        self.track_trajectories: Dict[int, deque] = {}
        
        # Estadísticas de rendering
        self.render_times = []
        self.total_renders = 0
        
        # Estado de la visualización
        self.show_fps = True
        self.show_stats = True
    
    def _load_config(self) -> None:
        """Carga la configuración del visualizador."""
        # Configuración de colores
        self.bbox_colors = get_config('visualizer.bbox_colors', [
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255]
        ])
        
        # Configuración de texto
        self.font_scale = get_config('visualizer.font_scale', 0.6)
        self.font_thickness = get_config('visualizer.font_thickness', 2)
        
        # Configuración de trayectorias
        self.show_trajectory = get_config('visualizer.show_trajectory', True)
        self.trajectory_length = get_config('visualizer.trajectory_length', 30)
        self.trajectory_thickness = get_config('visualizer.trajectory_thickness', 2)
        
        # Información a mostrar
        self.show_id = get_config('visualizer.show_id', True)
        self.show_class = get_config('visualizer.show_class', True)
        self.show_confidence = get_config('visualizer.show_confidence', True)
        self.show_velocity = get_config('visualizer.show_velocity', True)
        self.show_distance = get_config('visualizer.show_distance', True)
        
        print(f"🎨 Configuración del visualizador:")
        print(f"   🎯 Mostrar trayectorias: {self.show_trajectory}")
        print(f"   📏 Longitud trayectoria: {self.trajectory_length}")
        print(f"   📊 Mostrar velocidad: {self.show_velocity}")
    
    def render_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        detections: Optional[List[Detection]] = None,
        metrics: Optional[Dict[int, ObjectMetrics]] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Renderiza un frame completo con toda la información.
        
        Args:
            frame: Frame base
            tracks: Lista de tracks activos
            detections: Lista de detecciones (opcional)
            metrics: Métricas por track (opcional)
            extra_info: Información adicional a mostrar
            
        Returns:
            Frame renderizado
        """
        start_time = time.time()
        
        # Copiar frame para no modificar el original
        rendered_frame = frame.copy()
        
        # Renderizar detecciones si se proporcionan
        if detections:
            rendered_frame = self._render_detections(rendered_frame, detections)
        
        # Renderizar tracks
        rendered_frame = self._render_tracks(rendered_frame, tracks, metrics)
        
        # Renderizar trayectorias
        if self.show_trajectory:
            rendered_frame = self._render_trajectories(rendered_frame, tracks)
        
        # Renderizar información general
        rendered_frame = self._render_info_panel(rendered_frame, tracks, metrics, extra_info)
        
        # Estadísticas de rendering
        render_time = time.time() - start_time
        self.render_times.append(render_time)
        self.total_renders += 1
        
        # Mantener solo los últimos 100 tiempos
        if len(self.render_times) > 100:
            self.render_times = self.render_times[-100:]
        
        return rendered_frame
    
    def _render_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Renderiza detecciones básicas (sin tracking).
        
        Args:
            frame: Frame base
            detections: Lista de detecciones
            
        Returns:
            Frame con detecciones renderizadas
        """
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            
            # Color base para detecciones (gris)
            color = (128, 128, 128)
            
            # Dibujar bounding box con línea punteada
            self._draw_dashed_rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            # Etiqueta simple
            if self.show_class:
                label = f"{detection.class_name}"
                if self.show_confidence:
                    label += f": {detection.confidence:.2f}"
                
                self._draw_text_with_background(
                    frame, label, (x1, y1 - 5), color, scale=0.4
                )
        
        return frame
    
    def _render_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        metrics: Optional[Dict[int, ObjectMetrics]] = None
    ) -> np.ndarray:
        """
        Renderiza tracks con información completa.
        
        Args:
            frame: Frame base
            tracks: Lista de tracks
            metrics: Métricas asociadas
            
        Returns:
            Frame con tracks renderizados
        """
        for track in tracks:
            if not track.is_confirmed:
                continue
            
            x1, y1, x2, y2 = map(int, track.bbox)
            
            # Obtener color del track
            color = self._get_track_color(track.track_id)
            
            # Dibujar bounding box principal
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Preparar información del track
            info_lines = []
            
            if self.show_id:
                info_lines.append(f"ID: {track.track_id}")
            
            if self.show_class:
                info_lines.append(f"{track.class_name}")
            
            if self.show_confidence:
                info_lines.append(f"Conf: {track.confidence:.2f}")
            
            # Agregar métricas si están disponibles
            if metrics and track.track_id in metrics:
                track_metrics = metrics[track.track_id]
                
                if self.show_velocity:
                    speed_kmh = track_metrics.speed_meters_per_second * 3.6
                    info_lines.append(f"Vel: {speed_kmh:.1f} km/h")
                
                if self.show_distance:
                    info_lines.append(f"Dist: {track_metrics.distance_traveled_meters:.1f}m")
            
            # Renderizar información
            self._draw_track_info(frame, info_lines, (x1, y1), color)
            
            # Punto central del track
            center = track.center
            cv2.circle(frame, (int(center[0]), int(center[1])), 3, color, -1)
        
        return frame
    
    def _render_trajectories(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Renderiza las trayectorias de los tracks.
        
        Args:
            frame: Frame base
            tracks: Lista de tracks
            
        Returns:
            Frame con trayectorias renderizadas
        """
        for track in tracks:
            if not track.is_confirmed or len(track.trajectory) < 2:
                continue
            
            color = self._get_track_color(track.track_id)
            trajectory_points = list(track.trajectory)
            
            # Actualizar cache de trayectoria
            if track.track_id not in self.track_trajectories:
                self.track_trajectories[track.track_id] = deque(maxlen=self.trajectory_length)
            
            current_center = track.center
            self.track_trajectories[track.track_id].append(current_center)
            
            # Dibujar trayectoria
            points = list(self.track_trajectories[track.track_id])
            if len(points) >= 2:
                # Crear gradiente de opacidad
                for i in range(1, len(points)):
                    alpha = i / len(points)  # Más opaco hacia el final
                    
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    
                    # Color con alpha
                    line_color = tuple(int(c * alpha) for c in color)
                    
                    cv2.line(frame, pt1, pt2, line_color, self.trajectory_thickness)
        
        return frame
    
    def _render_info_panel(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        metrics: Optional[Dict[int, ObjectMetrics]] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Renderiza panel de información general.
        
        Args:
            frame: Frame base
            tracks: Lista de tracks
            metrics: Métricas del sistema
            extra_info: Información adicional
            
        Returns:
            Frame con panel de información
        """
        height, width = frame.shape[:2]
        panel_width = 250
        panel_height = 150
        
        # Crear panel semi-transparente
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (width - panel_width - 10, 10),
            (width - 10, panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Información a mostrar
        info_lines = []
        
        if self.show_stats:
            # Estadísticas básicas
            active_tracks = len([t for t in tracks if t.is_confirmed])
            info_lines.append(f"Tracks activos: {active_tracks}")
            
            # FPS de rendering
            if self.show_fps and len(self.render_times) > 0:
                avg_render_time = np.mean(self.render_times)
                render_fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
                info_lines.append(f"Render FPS: {render_fps:.1f}")
            
            # Métricas globales
            if metrics:
                speeds = [m.speed_meters_per_second for m in metrics.values()]
                if speeds:
                    avg_speed_kmh = np.mean(speeds) * 3.6
                    max_speed_kmh = np.max(speeds) * 3.6
                    info_lines.append(f"Vel. prom: {avg_speed_kmh:.1f} km/h")
                    info_lines.append(f"Vel. máx: {max_speed_kmh:.1f} km/h")
        
        # Información adicional
        if extra_info:
            for key, value in extra_info.items():
                info_lines.append(f"{key}: {value}")
        
        # Renderizar líneas de información
        y_offset = 30
        for line in info_lines:
            cv2.putText(
                frame,
                line,
                (width - panel_width + 5, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 20
        
        return frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Obtiene un color consistente para un track.
        
        Args:
            track_id: ID del track
            
        Returns:
            Color BGR
        """
        if track_id not in self.track_colors:
            # Asignar color de la paleta
            color_index = track_id % len(self.bbox_colors)
            self.track_colors[track_id] = tuple(self.bbox_colors[color_index])
        
        return self.track_colors[track_id]
    
    def _draw_dashed_rectangle(
        self,
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash_length: int = 5
    ) -> None:
        """
        Dibuja un rectángulo con líneas punteadas.
        
        Args:
            frame: Frame donde dibujar
            pt1: Punto superior izquierdo
            pt2: Punto inferior derecho
            color: Color BGR
            thickness: Grosor de línea
            dash_length: Longitud de cada segmento
        """
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Líneas horizontales
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Líneas verticales
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = None,
        bg_alpha: float = 0.7
    ) -> None:
        """obs
        Dibuja texto con fondo semi-transparente.
        
        Args:
            frame: Frame donde dibujar
            text: Texto a dibujar
            position: Posición (x, y)
            color: Color del texto
            scale: Escala del texto (None = usar configuración)
            bg_alpha: Transparencia del fondo
        """
        if scale is None:
            scale = self.font_scale
        
        # Obtener tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, self.font_thickness
        )
        
        x, y = position
        
        # Dibujar fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 2, y - text_height - baseline - 2),
            (x + text_width + 2, y + baseline + 2),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0, frame)
        
        # Dibujar texto
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            self.font_thickness
        )
    
    def _draw_track_info(
        self,
        frame: np.ndarray,
        info_lines: List[str],
        position: Tuple[int, int],
        color: Tuple[int, int, int]
    ) -> None:
        """
        Dibuja información del track en múltiples líneas.
        
        Args:
            frame: Frame donde dibujar
            info_lines: Lista de líneas de información
            position: Posición inicial
            color: Color del texto
        """
        x, y = position
        line_height = int(self.font_scale * 30)
        
        for i, line in enumerate(info_lines):
            line_y = y - (len(info_lines) - i) * line_height
            self._draw_text_with_background(frame, line, (x, line_y), color)
    
    def toggle_trajectory_display(self) -> None:
        """Alterna la visualización de trayectorias."""
        self.show_trajectory = not self.show_trajectory
        print(f"🎨 Trayectorias: {'ON' if self.show_trajectory else 'OFF'}")
    
    def toggle_fps_display(self) -> None:
        """Alterna la visualización de FPS."""
        self.show_fps = not self.show_fps
        print(f"🎨 FPS: {'ON' if self.show_fps else 'OFF'}")
    
    def toggle_stats_display(self) -> None:
        """Alterna la visualización de estadísticas."""
        self.show_stats = not self.show_stats
        print(f"🎨 Estadísticas: {'ON' if self.show_stats else 'OFF'}")
    
    def clear_trajectories(self) -> None:
        """Limpia todas las trayectorias visualizadas."""
        self.track_trajectories.clear()
        print("🧹 Trayectorias visuales limpiadas")
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rendering.
        
        Returns:
            Diccionario con estadísticas
        """
        avg_render_time = np.mean(self.render_times) if self.render_times else 0
        render_fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
        
        return {
            'total_renders': self.total_renders,
            'avg_render_time': avg_render_time,
            'render_fps': render_fps,
            'active_track_colors': len(self.track_colors),
            'trajectory_cache_size': len(self.track_trajectories)
        } 