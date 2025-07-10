"""
Grabador - Recorder
==================

Este módulo maneja la grabación directa de GIFs animados
para documentación y análisis de resultados.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import time
import threading
from pathlib import Path
from collections import deque

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("⚠️ imageio no está instalado. Ejecute: pip install imageio imageio-ffmpeg")

from ..utils.config_loader import get_config


class GifRecorder:
    """
    Grabador de GIFs animados optimizado.
    
    Permite capturar segmentos interesantes del procesamiento
    y exportarlos directamente como GIFs optimizados para documentación.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Inicializa el grabador de GIFs.
        
        Args:
            output_dir: Directorio de salida
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cargar configuración
        self._load_config()
        
        # Estado de grabación
        self.is_recording = False
        self.current_gif_path = None
        self.current_gif_filename = None
        
        # Buffer para grabación actual
        self.recording_buffer = deque(maxlen=int(self.gif_fps * self.max_gif_duration))
        self.auto_record_buffer = deque(maxlen=150)  # ~5 segundos a 30fps
        
        # Control de frame skipping
        self.frames_skipped = 0
        
        # Estadísticas
        self.gifs_created = 0
        self.total_frames_recorded = 0
        
        print(f"🎞️ Grabador de GIFs inicializado:")
        print(f"   📁 Directorio: {self.output_dir}")
        print(f"   🎬 Auto-grabado: {self.auto_record_scenes}")
        print(f"   ⚙️ Resolución GIF: {self.gif_width}x{self.gif_height}")
        print(f"   🎯 FPS: {self.gif_fps} | Calidad: {self.gif_quality}%")
    
    def _load_config(self) -> None:
        """Carga la configuración del grabador de GIFs."""
        # Configuración de GIFs
        self.gif_fps = get_config('recorder.gif_fps', 8)
        self.gif_quality = get_config('recorder.gif_quality', 75)
        self.gif_width = get_config('recorder.gif_width', 640)
        self.gif_height = get_config('recorder.gif_height', 480)
        
        # Duración máxima de GIFs (para evitar archivos muy grandes)
        self.max_gif_duration = get_config('recorder.max_gif_duration', 10.0)
        self.min_gif_duration = get_config('recorder.min_gif_duration', 2.0)
        
        # Auto-grabación
        self.auto_record_scenes = get_config('recorder.auto_record_interesting_scenes', True)
        self.min_objects_for_recording = get_config('recorder.min_objects_for_recording', 2)
        
        # Optimización para GIFs
        self.frame_skip = get_config('recorder.frame_skip', 3)  # Tomar 1 de cada N frames
        self.reduce_colors = get_config('recorder.reduce_colors', True)
        self.optimize_gif = get_config('recorder.optimize_gif', True)
    
    def start_recording(self, filename: Optional[str] = None) -> bool:
        """
        Inicia la grabación de GIF.
        
        Args:
            filename: Nombre del archivo (auto-generado si es None)
            
        Returns:
            True si se inició exitosamente
        """
        if not IMAGEIO_AVAILABLE:
            print("❌ imageio no está disponible para crear GIFs")
            return False
            
        if self.is_recording:
            print("⚠️ Ya hay una grabación en curso")
            return False
        
        # Generar nombre de archivo
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tracking_{timestamp}.gif"
        
        # Asegurar extensión .gif
        if not filename.endswith('.gif'):
            filename = filename.rsplit('.', 1)[0] + '.gif'
        
        self.current_gif_filename = filename
        self.current_gif_path = self.output_dir / filename
        
        # Limpiar buffer de grabación
        self.recording_buffer.clear()
        self.is_recording = True
        self.frames_skipped = 0
        
        print(f"🎞️ Grabación de GIF iniciada: {filename}")
        print(f"   📏 Resolución: {self.gif_width}x{self.gif_height}")
        print(f"   ⏱️ FPS: {self.gif_fps}")
        return True
    
    def record_frame(self, frame: np.ndarray) -> None:
        """
        Graba un frame al GIF actual.
        
        Args:
            frame: Frame a grabar
        """
        if not self.is_recording or not IMAGEIO_AVAILABLE:
            return
        
        # Aplicar frame skipping para optimizar el GIF
        self.frames_skipped += 1
        if self.frames_skipped < self.frame_skip:
            return
        
        self.frames_skipped = 0
        
        # Procesar frame para GIF
        processed_frame = self._process_frame_for_gif(frame)
        
        # Agregar al buffer de grabación
        self.recording_buffer.append(processed_frame)
        self.total_frames_recorded += 1
        
        # Si el buffer está lleno, detener grabación automáticamente
        max_frames = int(self.gif_fps * self.max_gif_duration)
        if len(self.recording_buffer) >= max_frames:
            print(f"⚠️ Duración máxima alcanzada ({self.max_gif_duration}s), finalizando grabación...")
            self.stop_recording()
    
    def _process_frame_for_gif(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame para optimizarlo para GIF.
        
        Args:
            frame: Frame original
            
        Returns:
            Frame procesado
        """
        # Redimensionar a resolución del GIF
        processed_frame = cv2.resize(frame, (self.gif_width, self.gif_height))
        
        # Reducir colores si está habilitado
        if self.reduce_colors:
            # Convertir a espacio de color más limitado para reducir tamaño
            processed_frame = cv2.convertScaleAbs(processed_frame, alpha=0.9, beta=10)
        
        return processed_frame
    
    def stop_recording(self) -> Optional[str]:
        """
        Detiene la grabación de GIF y guarda el archivo.
        
        Returns:
            Ruta del archivo creado o None si hubo error
        """
        if not self.is_recording or not IMAGEIO_AVAILABLE:
            return None
        
        self.is_recording = False
        
        # Verificar duración mínima
        duration = len(self.recording_buffer) / self.gif_fps
        if duration < self.min_gif_duration:
            print(f"⚠️ Grabación muy corta ({duration:.1f}s), mínimo {self.min_gif_duration}s")
            return None
        
        # Verificar que hay frames
        if len(self.recording_buffer) == 0:
            print("⚠️ No hay frames para guardar")
            return None
        
        try:
            # Convertir frames a RGB
            frames_rgb = []
            for frame in self.recording_buffer:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
            
            # Crear GIF con optimizaciones
            imageio.mimsave(
                str(self.current_gif_path),
                frames_rgb,
                fps=self.gif_fps,
                quantizer='nq' if self.optimize_gif else 'wu',
                palettesize=256 if self.reduce_colors else 256,
                subrectangles=self.optimize_gif
            )
            
            self.gifs_created += 1
            file_size = self.current_gif_path.stat().st_size / 1024 / 1024  # MB
            
            print(f"✅ GIF guardado: {self.current_gif_filename}")
            print(f"   📊 {len(self.recording_buffer)} frames | {duration:.1f}s | {file_size:.1f}MB")
            
            return str(self.current_gif_path)
            
        except Exception as e:
            print(f"❌ Error guardando GIF: {e}")
            return None
    
    def create_gif_from_frames(self, frames: List[np.ndarray], filename: Optional[str] = None) -> Optional[str]:
        """
        Crea un GIF directamente desde una lista de frames.
        
        Args:
            frames: Lista de frames
            filename: Nombre del archivo (auto-generado si es None)
            
        Returns:
            Ruta del GIF creado o None si hubo error
        """
        if not IMAGEIO_AVAILABLE:
            print("❌ imageio no disponible para crear GIFs")
            return None
        
        if not frames:
            print("⚠️ Lista de frames vacía")
            return None
        
        # Generar nombre de archivo
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"custom_gif_{timestamp}.gif"
        
        # Asegurar extensión .gif
        if not filename.endswith('.gif'):
            filename = filename.rsplit('.', 1)[0] + '.gif'
        
        gif_path = self.output_dir / filename
        
        try:
            # Procesar y convertir frames
            frames_rgb = []
            for frame in frames:
                processed_frame = self._process_frame_for_gif(frame)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
            
            # Crear GIF
            imageio.mimsave(
                str(gif_path),
                frames_rgb,
                fps=self.gif_fps,
                quantizer='nq' if self.optimize_gif else 'wu',
                palettesize=256 if self.reduce_colors else 256,
                subrectangles=self.optimize_gif
            )
            
            self.gifs_created += 1
            file_size = gif_path.stat().st_size / 1024 / 1024  # MB
            
            print(f"🎞️ GIF personalizado creado: {filename}")
            print(f"   📊 {len(frames)} frames | {file_size:.1f}MB")
            
            return str(gif_path)
            
        except Exception as e:
            print(f"❌ Error creando GIF personalizado: {e}")
            return None
    
    def auto_record_frame(self, frame: np.ndarray, num_objects: int) -> None:
        """
        Procesa un frame para auto-grabación basada en actividad.
        
        Args:
            frame: Frame a procesar
            num_objects: Número de objetos detectados
        """
        if not self.auto_record_scenes:
            return
        
        # Agregar frame al buffer de auto-grabación
        self.auto_record_buffer.append((frame.copy(), num_objects, time.time()))
        
        # Verificar si debe iniciar grabación automática
        if not self.is_recording and num_objects >= self.min_objects_for_recording:
            self._start_auto_recording()
        
        # Si está grabando, continuar hasta que baje la actividad
        elif self.is_recording and num_objects < self.min_objects_for_recording:
            # Contar frames consecutivos con poca actividad
            recent_activity = [item[1] for item in list(self.auto_record_buffer)[-10:]]
            if all(activity < self.min_objects_for_recording for activity in recent_activity):
                self._stop_auto_recording()
    
    def _start_auto_recording(self) -> None:
        """Inicia grabación automática."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"auto_recording_{timestamp}.gif"
        
        if self.start_recording(filename):
            print("🤖 Auto-grabación iniciada")
            
            # Grabar frames del buffer previo
            for frame_data in list(self.auto_record_buffer)[-30:]:  # Últimos 30 frames
                frame, _, _ = frame_data
                self.record_frame(frame)
    
    def _stop_auto_recording(self) -> None:
        """Detiene grabación automática."""
        gif_path = self.stop_recording()
        if gif_path:
            print("🤖 Auto-grabación completada")
    
    def create_highlights_gif(self, source_gifs: List[str], output_name: Optional[str] = None) -> Optional[str]:
        """
        Crea un GIF resumen con highlights de múltiples GIFs.
        
        Args:
            source_gifs: Lista de rutas de GIFs fuente
            output_name: Nombre del archivo de salida
            
        Returns:
            Ruta del GIF resumen o None si hubo error
        """
        if not IMAGEIO_AVAILABLE or not source_gifs:
            return None
        
        if output_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"highlights_{timestamp}.gif"
        
        # Asegurar extensión .gif
        if not output_name.endswith('.gif'):
            output_name = output_name.rsplit('.', 1)[0] + '.gif'
        
        output_path = self.output_dir / output_name
        
        try:
            all_frames = []
            
            # Leer frames de cada GIF fuente
            for gif_path in source_gifs:
                if not Path(gif_path).exists():
                    print(f"⚠️ GIF no encontrado: {gif_path}")
                    continue
                
                try:
                    gif_frames = imageio.mimread(gif_path)
                    # Tomar solo algunos frames de cada GIF para el resumen
                    sample_frames = gif_frames[::max(1, len(gif_frames) // 10)]  # Máximo 10 frames por GIF
                    all_frames.extend(sample_frames)
                except Exception as e:
                    print(f"⚠️ Error leyendo {gif_path}: {e}")
                    continue
            
            if not all_frames:
                print("❌ No se pudieron leer frames de los GIFs fuente")
                return None
            
            # Crear GIF resumen
            imageio.mimsave(
                str(output_path),
                all_frames,
                fps=self.gif_fps,
                quantizer='nq' if self.optimize_gif else 'wu',
                palettesize=256,
                subrectangles=self.optimize_gif
            )
            
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            print(f"🎞️ GIF de highlights creado: {output_name}")
            print(f"   📊 {len(all_frames)} frames | {file_size:.1f}MB")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error creando GIF de highlights: {e}")
            return None
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del grabador.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'gifs_created': self.gifs_created,
            'total_frames_recorded': self.total_frames_recorded,
            'is_recording': self.is_recording,
            'current_buffer_size': len(self.recording_buffer) if self.is_recording else 0,
            'auto_record_enabled': self.auto_record_scenes,
            'output_directory': str(self.output_dir),
            'gif_settings': {
                'fps': self.gif_fps,
                'resolution': f"{self.gif_width}x{self.gif_height}",
                'quality': self.gif_quality,
                'optimize': self.optimize_gif,
                'reduce_colors': self.reduce_colors
            }
        }
    
    def cleanup(self) -> None:
        """Limpia recursos del grabador."""
        if self.is_recording:
            print("🧹 Finalizando grabación en curso...")
            self.stop_recording()
        
        self.recording_buffer.clear()
        self.auto_record_buffer.clear()
        
        print("🧹 Grabador limpiado")


# Alias para compatibilidad hacia atrás
Recorder = GifRecorder 