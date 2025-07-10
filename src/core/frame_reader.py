"""
Lector de Frames - FrameReader
=============================

Este módulo maneja la lectura de frames desde videos o cámaras
en tiempo real, con soporte para redimensionamiento y optimización.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator, Union
from pathlib import Path
import time

from ..utils.config_loader import get_config


class FrameReader:
    """
    Lector de frames desde video o cámara.
    
    Proporciona una interfaz unificada para leer frames desde
    diferentes fuentes (archivos de video, cámara web, etc.)
    """
    
    def __init__(self, source: Union[str, int, None] = None):
        """
        Inicializa el lector de frames.
        
        Args:
            source: Fuente de video. Puede ser:
                   - None: usa configuración por defecto
                   - int: ID de cámara (0, 1, 2...)
                   - str: ruta a archivo de video
        """
        self.source = source
        self.cap = None
        self.is_camera = False
        self.frame_count = 0
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
        
        # Cargar configuración
        self._load_config()
        
        # Determinar el tipo de fuente
        self._determine_source()
        
        # Estadísticas
        self.start_time = None
        self.frames_processed = 0
        
    def _load_config(self) -> None:
        """Carga la configuración del frame reader."""
        self.resize_frame = get_config('frame_reader.resize_frame', True)
        self.frame_width = get_config('frame_reader.frame_width', 1280)
        self.frame_height = get_config('frame_reader.frame_height', 720)
        self.skip_frames = get_config('frame_reader.skip_frames', 0)
        
        if self.source is None:
            # Usar configuración por defecto
            camera_id = get_config('frame_reader.camera_id', 0)
            video_path = get_config('frame_reader.video_path', '')
            
            # Preferir cámara si no hay video configurado
            if video_path and Path(video_path).exists():
                self.source = video_path
            else:
                self.source = camera_id
    
    def _determine_source(self) -> None:
        """Determina el tipo de fuente (cámara o video)."""
        if isinstance(self.source, int):
            self.is_camera = True
        elif isinstance(self.source, str):
            self.is_camera = False
            if not Path(self.source).exists():
                print(f"⚠️ Archivo de video no encontrado: {self.source}")
                print("🔄 Cambiando a cámara por defecto...")
                self.source = 0
                self.is_camera = True
        else:
            # Por defecto usar cámara
            self.source = 0
            self.is_camera = True
    
    def initialize(self) -> bool:
        """
        Inicializa la captura de video.
        
        Returns:
            True si la inicialización fue exitosa, False en caso contrario
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                print(f"❌ Error: No se pudo abrir la fuente: {self.source}")
                return False
            
            # Configurar parámetros de captura
            if self.is_camera:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                print(f"📹 Cámara inicializada (ID: {self.source})")
            else:
                # Para archivos de video, obtener propiedades
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / self.fps if self.fps > 0 else 0
                
                print(f"🎬 Video cargado: {self.source}")
                print(f"   📊 FPS: {self.fps:.2f}")
                print(f"   🎞️ Frames totales: {total_frames}")
                print(f"   ⏱️ Duración: {duration:.2f}s")
            
            # Obtener dimensiones reales
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"   🖼️ Resolución: {actual_width}x{actual_height}")
            
            self.start_time = time.time()
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando captura: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee un frame de la fuente.
        
        Returns:
            Tupla (success, frame) donde success indica si la lectura
            fue exitosa y frame es el array numpy del frame
        """
        if self.cap is None:
            return False, None
        
        # Saltar frames si está configurado
        for _ in range(self.skip_frames):
            ret, _ = self.cap.read()
            if not ret:
                return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            return False, None
        
        # Redimensionar si está habilitado
        if self.resize_frame and frame is not None:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        self.frames_processed += 1
        return True, frame
    
    def frame_generator(self) -> Generator[np.ndarray, None, None]:
        """
        Generador que produce frames continuamente.
        
        Yields:
            Arrays numpy de frames
        """
        while True:
            ret, frame = self.read_frame()
            if not ret:
                if self.is_camera:
                    # Para cámaras, intentar reconectar
                    time.sleep(0.1)
                    continue
                else:
                    # Para videos, terminar cuando se acabe
                    break
            
            yield frame
    
    def get_fps(self) -> float:
        """
        Obtiene los FPS de la fuente.
        
        Returns:
            FPS de la fuente de video
        """
        if self.cap is not None:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return self.fps
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Obtiene el tamaño del frame.
        
        Returns:
            Tupla (width, height) del frame
        """
        if self.resize_frame:
            return self.frame_width, self.frame_height
        elif self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 640, 480
    
    def get_total_frames(self) -> int:
        """
        Obtiene el número total de frames (solo para videos).
        
        Returns:
            Número total de frames, -1 para cámaras
        """
        if self.is_camera:
            return -1
        elif self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0
    
    def get_current_frame_number(self) -> int:
        """
        Obtiene el número del frame actual.
        
        Returns:
            Número del frame actual
        """
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return 0
    
    def seek_frame(self, frame_number: int) -> bool:
        """
        Busca un frame específico (solo para videos).
        
        Args:
            frame_number: Número del frame a buscar
            
        Returns:
            True si fue exitoso, False en caso contrario
        """
        if self.is_camera:
            return False
        
        if self.cap is not None:
            return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return False
    
    def get_processing_stats(self) -> dict:
        """
        Obtiene estadísticas de procesamiento.
        
        Returns:
            Diccionario con estadísticas
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        fps_actual = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed_time,
            'fps_actual': fps_actual,
            'fps_source': self.get_fps(),
            'source_type': 'camera' if self.is_camera else 'video',
            'source': str(self.source)
        }
    
    def release(self) -> None:
        """Libera los recursos de captura."""
        if self.cap is not None:
            self.cap.release()
            print("📴 Captura de video liberada")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release() 