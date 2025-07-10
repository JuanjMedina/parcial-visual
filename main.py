#!/usr/bin/env python3
"""
Sistema de Detección y Seguimiento de Objetos en Tiempo Real
============================================================

Aplicación principal que integra:
- Detección con YOLO
- Seguimiento con DeepSORT
- Cálculo de métricas
- Visualización en tiempo real
- Grabación de GIFs

Autor: Parcial Visual Computing
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import threading
from typing import Optional

# Importar módulos del sistema
from src.core.frame_reader import FrameReader
from src.core.detector import YOLODetector
from src.core.tracker import ObjectTracker
from src.core.metrics_calculator import MetricsCalculator
from src.core.visualizer import Visualizer
from src.core.recorder import Recorder
from src.utils.config_loader import config, get_config


class ObjectTrackingSystem:
    """
    Sistema principal de detección y seguimiento de objetos.
    
    Integra todos los módulos en un pipeline completo de procesamiento
    en tiempo real con visualización y grabación.
    """
    
    def __init__(self, source: Optional[str] = None, output_dir: str = "output"):
        """
        Inicializa el sistema completo.
        
        Args:
            source: Fuente de video (None = usar configuración)
            output_dir: Directorio de salida
        """
        print("🚀 Inicializando Sistema de Detección y Seguimiento...")
        
        # Crear directorio de salida
        Path(output_dir).mkdir(exist_ok=True)
        
        # Inicializar componentes
        self.frame_reader = FrameReader(source)
        self.detector = YOLODetector()
        self.tracker = ObjectTracker()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        self.recorder = Recorder(output_dir)
        
        # Estado del sistema
        self.is_running = False
        self.paused = False
        self.frame_count = 0
        self.start_time = None
        
        # Estadísticas
        self.total_detections = 0
        self.total_tracks = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        print("✅ Sistema inicializado correctamente")
    
    def run(self, headless: bool = False, record_video: bool = False, create_gifs: bool = True) -> None:
        """
        Ejecuta el sistema principal.
        
        Args:
            headless: Ejecutar sin interfaz gráfica
            record_video: Grabar video automáticamente
            create_gifs: Crear GIFs automáticamente
        """
        print("🎬 Iniciando procesamiento...")
        
        # Inicializar frame reader
        if not self.frame_reader.initialize():
            print("❌ Error inicializando fuente de video")
            return
        
        # Iniciar grabación de GIF si está habilitado
        if record_video:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.recorder.start_recording(f"session_{timestamp}.gif")
        
        self.is_running = True
        self.start_time = time.time()
        
        try:
            # Loop principal de procesamiento
            for frame in self.frame_reader.frame_generator():
                if not self.is_running:
                    break
                
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Procesar frame
                processed_frame = self._process_frame(frame)
                
                # Mostrar frame si no es headless
                if not headless:
                    cv2.imshow('Object Tracking System', processed_frame)
                    
                    # Controles de teclado
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        self.paused = not self.paused
                        print(f"{'⏸️ Pausado' if self.paused else '▶️ Reanudado'}")
                    elif key == ord('r'):
                        if not self.recorder.is_recording:
                            timestamp = time.strftime("%H%M%S")
                            self.recorder.start_recording(f"manual_{timestamp}.gif")
                            print("🔴 Grabación manual de GIF iniciada")
                        else:
                            gif_path = self.recorder.stop_recording()
                            if gif_path:
                                print(f"⏹️ GIF guardado: {gif_path}")
                    elif key == ord('g'):
                        # Crear GIF desde frames del buffer automático
                        if hasattr(self.recorder, 'auto_record_buffer') and self.recorder.auto_record_buffer:
                            frames = [item[0] for item in list(self.recorder.auto_record_buffer)]
                            gif_path = self.recorder.create_gif_from_frames(frames)
                            if gif_path:
                                print(f"🎞️ GIF instantáneo creado: {gif_path}")
                    elif key == ord('s'):
                        self._print_statistics()
                
                # Grabar frame si está habilitado
                if record_video or self.recorder.is_recording:
                    self.recorder.record_frame(processed_frame)
                
                # Auto-grabación basada en actividad
                if create_gifs:
                    num_objects = len([t for t in getattr(self, 'current_tracks', []) if t.is_confirmed])
                    self.recorder.auto_record_frame(processed_frame, num_objects)
                
                self.frame_count += 1
                self._update_fps()
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción de usuario")
        
        finally:
            self._cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Procesa un frame completo a través del pipeline.
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Frame procesado con visualizaciones
        """
        # 1. Detección de objetos
        detections = self.detector.detect(frame)
        self.total_detections += len(detections)
        
        # 2. Seguimiento de objetos
        tracks = self.tracker.update(detections, frame)
        self.current_tracks = tracks  # Guardar para auto-grabación
        active_tracks = [t for t in tracks if t.is_confirmed]
        self.total_tracks = len(active_tracks)
        
        # 3. Cálculo de métricas
        metrics = self.metrics_calculator.calculate_metrics(active_tracks)
        
        # 4. Visualización
        extra_info = {
            'FPS': f"{self.current_fps:.1f}",
            'Frame': self.frame_count,
            'Detecciones': len(detections),
            'Tiempo': f"{time.time() - self.start_time:.1f}s" if self.start_time else "0s"
        }
        
        processed_frame = self.visualizer.render_frame(
            frame=frame,
            tracks=active_tracks,
            detections=detections,
            metrics=metrics,
            extra_info=extra_info
        )
        
        return processed_frame
    
    def _update_fps(self) -> None:
        """Actualiza el contador de FPS."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _print_statistics(self) -> None:
        """Imprime estadísticas del sistema."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n📊 Estadísticas del Sistema:")
        print("=" * 50)
        print(f"⏱️ Tiempo transcurrido: {elapsed_time:.1f}s")
        print(f"🎞️ Frames procesados: {self.frame_count}")
        print(f"📹 FPS actual: {self.current_fps:.1f}")
        print(f"🎯 Total detecciones: {self.total_detections}")
        print(f"👥 Tracks activos: {self.total_tracks}")
        
        # Estadísticas de módulos
        detector_stats = self.detector.get_statistics()
        print(f"🔍 FPS detección: {detector_stats['fps']:.1f}")
        
        tracker_stats = self.tracker.get_statistics()
        print(f"🎯 Tipo tracker: {tracker_stats['tracker_type']}")
        print(f"📈 Total tracks: {tracker_stats['total_tracks']}")
        
        metrics_stats = self.metrics_calculator.get_global_statistics()
        print(f"🚀 Velocidad promedio: {metrics_stats['avg_speed_mps'] * 3.6:.1f} km/h")
        print(f"📏 Distancia total: {metrics_stats['total_distance_m']:.1f}m")
        
        recorder_stats = self.recorder.get_statistics()
        print(f"🎞️ GIFs creados: {recorder_stats['gifs_created']}")
        print(f"📊 Frames totales grabados: {recorder_stats['total_frames_recorded']}")
        print("=" * 50)
    
    def _cleanup(self) -> None:
        """Limpia recursos del sistema."""
        print("\n🧹 Limpiando recursos...")
        
        self.is_running = False
        
        # Detener grabación si está activa
        if self.recorder.is_recording:
            gif_path = self.recorder.stop_recording()
            if gif_path:
                print(f"💾 GIF final guardado: {gif_path}")
        
        # Crear GIF final del buffer automático si hay contenido
        if hasattr(self.recorder, 'auto_record_buffer') and self.recorder.auto_record_buffer:
            frames = [item[0] for item in list(self.recorder.auto_record_buffer)]
            final_gif = self.recorder.create_gif_from_frames(frames, "session_final.gif")
            if final_gif:
                print(f"🎞️ GIF final creado: {final_gif}")
        
        # Liberar recursos
        self.frame_reader.release()
        self.recorder.cleanup()
        cv2.destroyAllWindows()
        
        # Estadísticas finales
        self._print_statistics()
        
        print("✅ Sistema cerrado correctamente")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Sistema de Detección y Seguimiento de Objetos')
    
    # Argumentos principales
    parser.add_argument('--source', '-s', type=str, help='Fuente de video (ruta o número de cámara)')
    parser.add_argument('--output', '-o', type=str, default='output', help='Directorio de salida')
    parser.add_argument('--headless', action='store_true', help='Ejecutar sin interfaz gráfica')
    parser.add_argument('--record', '-r', action='store_true', help='Grabar video automáticamente')
    parser.add_argument('--no-gifs', action='store_true', help='Deshabilitar creación automática de GIFs')
    
    # Argumentos de configuración
    parser.add_argument('--model', type=str, help='Ruta al modelo YOLO')
    parser.add_argument('--confidence', type=float, help='Umbral de confianza (0-1)')
    parser.add_argument('--device', type=str, help='Dispositivo (cpu/cuda)')
    parser.add_argument('--pixels-per-meter', type=float, help='Calibración píxeles por metro')
    
    args = parser.parse_args()
    
    try:
        # Actualizar configuración si se especificaron argumentos
        if args.model:
            config.update('detector.model_path', args.model)
        if args.confidence:
            config.update('detector.confidence_threshold', args.confidence)
        if args.device:
            config.update('detector.device', args.device)
        if args.pixels_per_meter:
            config.update('metrics.pixels_per_meter', args.pixels_per_meter)
        
        # Preparar source (mantener como string, se manejará internamente)
        source = args.source
        
        # Crear y ejecutar sistema
        system = ObjectTrackingSystem(source=source, output_dir=args.output)
        
        print("\n🎮 Controles:")
        print("  Q - Salir")
        print("  ESPACIO - Pausar/Reanudar")
        print("  R - Iniciar/Detener grabación manual")
        print("  G - Crear GIF del buffer actual")
        print("  S - Mostrar estadísticas")
        print()
        
        system.run(
            headless=args.headless,
            record_video=args.record,
            create_gifs=not args.no_gifs
        )
        
    except Exception as e:
        print(f"❌ Error en el sistema: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 