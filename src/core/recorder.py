"""
Grabador - Recorder
==================

Este módulo maneja la grabación de videos, creación de GIFs
y exportación de resultados para documentación.
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


class Recorder:
    """
    Grabador de video y creador de GIFs.
    
    Permite capturar segmentos interesantes del procesamiento
    y exportarlos como videos o GIFs para documentación.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Inicializa el grabador.
        
        Args:
            output_dir: Directorio de salida
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Cargar configuración
        self._load_config()
        
        # Estado de grabación
        self.is_recording = False
        self.current_video_writer = None
        self.current_video_path = None
        
        # Buffer para GIFs
        self.gif_buffer = deque(maxlen=int(self.gif_fps * self.gif_duration))
        self.auto_record_buffer = deque(maxlen=150)  # ~5 segundos a 30fps
        
        # Estadísticas
        self.videos_created = 0
        self.gifs_created = 0
        self.total_frames_recorded = 0
        
        print(f"📹 Grabador inicializado:")
        print(f"   📁 Directorio: {self.output_dir}")
        print(f"   🎬 Auto-grabado: {self.auto_record_scenes}")
    
    def _load_config(self) -> None:
        """Carga la configuración del grabador."""
        self.gif_duration = get_config('recorder.gif_duration', 5.0)
        self.gif_fps = get_config('recorder.gif_fps', 10)
        self.gif_quality = get_config('recorder.gif_quality', 80)
        
        self.output_fps = get_config('recorder.output_fps', 30)
        self.output_codec = get_config('recorder.output_codec', 'mp4v')
        
        self.auto_record_scenes = get_config('recorder.auto_record_interesting_scenes', True)
        self.min_objects_for_recording = get_config('recorder.min_objects_for_recording', 2)
    
    def start_video_recording(self, filename: Optional[str] = None, frame_size: Optional[Tuple[int, int]] = None) -> bool:
        """
        Inicia la grabación de video.
        
        Args:
            filename: Nombre del archivo (auto-generado si es None)
            frame_size: Tamaño del frame (detectado automáticamente si es None)
            
        Returns:
            True si se inició exitosamente
        """
        if self.is_recording:
            print("⚠️ Ya hay una grabación en curso")
            return False
        
        # Generar nombre de archivo
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tracking_video_{timestamp}.mp4"
        
        self.current_video_path = self.output_dir / filename
        
        # Configurar VideoWriter (se inicializará con el primer frame)
        self.frame_size = frame_size
        self.is_recording = True
        
        print(f"🎬 Grabación iniciada: {filename}")
        return True
    
    def record_frame(self, frame: np.ndarray) -> None:
        """
        Graba un frame al video actual.
        
        Args:
            frame: Frame a grabar
        """
        if not self.is_recording:
            return
        
        # Inicializar VideoWriter si es necesario
        if self.current_video_writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
            
            self.current_video_writer = cv2.VideoWriter(
                str(self.current_video_path),
                fourcc,
                self.output_fps,
                (width, height)
            )
            
            if not self.current_video_writer.isOpened():
                print(f"❌ Error inicializando VideoWriter")
                self.stop_video_recording()
                return
        
        # Escribir frame
        self.current_video_writer.write(frame)
        self.total_frames_recorded += 1
        
        # Agregar al buffer de GIF
        if len(self.gif_buffer) == 0 or len(self.gif_buffer) % (30 // self.gif_fps) == 0:
            # Reducir resolución para GIF
            gif_frame = cv2.resize(frame, (640, 360))
            self.gif_buffer.append(gif_frame)
    
    def stop_video_recording(self) -> Optional[str]:
        """
        Detiene la grabación de video.
        
        Returns:
            Ruta del archivo creado o None si hubo error
        """
        if not self.is_recording:
            return None
        
        self.is_recording = False
        
        if self.current_video_writer is not None:
            self.current_video_writer.release()
            self.current_video_writer = None
            
            self.videos_created += 1
            print(f"✅ Video guardado: {self.current_video_path}")
            
            return str(self.current_video_path)
        
        return None
    
    def create_gif_from_buffer(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Crea un GIF del buffer actual.
        
        Args:
            filename: Nombre del archivo (auto-generado si es None)
            
        Returns:
            Ruta del GIF creado o None si hubo error
        """
        if not IMAGEIO_AVAILABLE:
            print("❌ imageio no disponible para crear GIFs")
            return None
        
        if len(self.gif_buffer) == 0:
            print("⚠️ Buffer de GIF vacío")
            return None
        
        # Generar nombre de archivo
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"tracking_gif_{timestamp}.gif"
        
        gif_path = self.output_dir / filename
        
        try:
            # Convertir frames a RGB
            frames_rgb = []
            for frame in self.gif_buffer:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
            
            # Crear GIF
            imageio.mimsave(
                str(gif_path),
                frames_rgb,
                fps=self.gif_fps,
                quality=self.gif_quality
            )
            
            self.gifs_created += 1
            print(f"🎞️ GIF creado: {filename}")
            return str(gif_path)
            
        except Exception as e:
            print(f"❌ Error creando GIF: {e}")
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
        filename = f"auto_recording_{timestamp}.mp4"
        
        if self.start_video_recording(filename):
            print("🤖 Auto-grabación iniciada")
            
            # Grabar frames del buffer previo
            for frame_data in list(self.auto_record_buffer)[-30:]:  # Últimos 30 frames
                frame, _, _ = frame_data
                self.record_frame(frame)
    
    def _stop_auto_recording(self) -> None:
        """Detiene grabación automática."""
        video_path = self.stop_video_recording()
        if video_path:
            print("🤖 Auto-grabación completada")
            
            # Crear GIF automáticamente
            threading.Thread(
                target=self._create_auto_gif,
                args=(video_path,),
                daemon=True
            ).start()
    
    def _create_auto_gif(self, video_path: str) -> None:
        """
        Crea un GIF automáticamente de un video.
        
        Args:
            video_path: Ruta del video
        """
        try:
            # Obtener nombre base sin extensión
            video_name = Path(video_path).stem
            gif_name = f"{video_name}.gif"
            
            self.create_gif_from_buffer(gif_name)
            
        except Exception as e:
            print(f"❌ Error creando GIF automático: {e}")
    
    def create_gif_from_video(self, video_path: str, start_time: float = 0, duration: Optional[float] = None) -> Optional[str]:
        """
        Crea un GIF desde un archivo de video existente.
        
        Args:
            video_path: Ruta del video
            start_time: Tiempo de inicio en segundos
            duration: Duración en segundos (None = hasta el final)
            
        Returns:
            Ruta del GIF creado o None si hubo error
        """
        if not IMAGEIO_AVAILABLE:
            print("❌ imageio no disponible")
            return None
        
        try:
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ No se pudo abrir el video: {video_path}")
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calcular frames a extraer
            start_frame = int(start_time * fps)
            if duration is not None:
                end_frame = int((start_time + duration) * fps)
            else:
                end_frame = total_frames
            
            # Extraer frames
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_step = max(1, int(fps / self.gif_fps))
            
            for frame_num in range(start_frame, min(end_frame, total_frames), frame_step):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Redimensionar y convertir a RGB
                frame_resized = cv2.resize(frame, (640, 360))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if not frames:
                print("⚠️ No se extrajeron frames del video")
                return None
            
            # Crear GIF
            video_name = Path(video_path).stem
            gif_name = f"{video_name}_segment.gif"
            gif_path = self.output_dir / gif_name
            
            imageio.mimsave(
                str(gif_path),
                frames,
                fps=self.gif_fps,
                quality=self.gif_quality
            )
            
            print(f"🎞️ GIF creado desde video: {gif_name}")
            return str(gif_path)
            
        except Exception as e:
            print(f"❌ Error creando GIF desde video: {e}")
            return None
    
    def create_highlights_reel(self, video_paths: List[str], output_name: Optional[str] = None) -> Optional[str]:
        """
        Crea un video resumen con los highlights.
        
        Args:
            video_paths: Lista de rutas de videos
            output_name: Nombre del archivo de salida
            
        Returns:
            Ruta del video resumen o None si hubo error
        """
        if not video_paths:
            return None
        
        if output_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_name = f"highlights_reel_{timestamp}.mp4"
        
        output_path = self.output_dir / output_name
        
        try:
            # Obtener propiedades del primer video
            first_cap = cv2.VideoCapture(video_paths[0])
            width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first_cap.release()
            
            # Crear writer
            fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, self.output_fps, (width, height))
            
            # Procesar cada video
            for video_path in video_paths:
                cap = cv2.VideoCapture(video_path)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    writer.write(frame)
                
                cap.release()
            
            writer.release()
            print(f"🎬 Video resumen creado: {output_name}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Error creando video resumen: {e}")
            return None
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del grabador.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'videos_created': self.videos_created,
            'gifs_created': self.gifs_created,
            'total_frames_recorded': self.total_frames_recorded,
            'is_recording': self.is_recording,
            'auto_record_enabled': self.auto_record_scenes,
            'gif_buffer_size': len(self.gif_buffer),
            'output_directory': str(self.output_dir)
        }
    
    def cleanup(self) -> None:
        """Limpia recursos y detiene grabaciones."""
        if self.is_recording:
            self.stop_video_recording()
        
        self.gif_buffer.clear()
        self.auto_record_buffer.clear()
        print("🧹 Grabador limpiado") 