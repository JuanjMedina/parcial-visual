#!/usr/bin/env python3
"""
Ejemplo Básico de Uso del Sistema de Tracking
============================================

Este archivo muestra cómo usar los módulos del sistema
de forma individual y combinada.
"""

import sys
import time
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.frame_reader import FrameReader
from src.core.detector import YOLODetector
from src.core.tracker import ObjectTracker
from src.core.metrics_calculator import MetricsCalculator
from src.core.visualizer import Visualizer
from src.core.recorder import Recorder


def ejemplo_deteccion_simple():
    """Ejemplo de detección simple sin tracking."""
    print("🔍 Ejemplo: Detección Simple")
    print("=" * 40)
    
    # Inicializar componentes
    frame_reader = FrameReader(0)  # Cámara
    detector = YOLODetector()
    
    if not frame_reader.initialize():
        print("❌ Error inicializando cámara")
        return
    
    print("📹 Presione 'q' para salir")
    
    try:
        for frame in frame_reader.frame_generator():
            # Detectar objetos
            detections = detector.detect(frame)
            
            # Dibujar detecciones
            frame_with_detections = detector._draw_detections(frame, detections)
            
            # Mostrar información
            print(f"Frame procesado: {len(detections)} detecciones")
            
            # Mostrar frame (opcional)
            import cv2
            cv2.imshow('Detección Simple', frame_with_detections)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        frame_reader.release()
        import cv2
        cv2.destroyAllWindows()


def ejemplo_tracking_basico():
    """Ejemplo de tracking básico con métricas."""
    print("🎯 Ejemplo: Tracking Básico")
    print("=" * 40)
    
    # Inicializar sistema completo
    frame_reader = FrameReader(0)  # Cámara
    detector = YOLODetector()
    tracker = ObjectTracker()
    metrics_calculator = MetricsCalculator()
    visualizer = Visualizer()
    
    if not frame_reader.initialize():
        print("❌ Error inicializando cámara")
        return
    
    print("📹 Controles:")
    print("  Q - Salir")
    print("  S - Mostrar estadísticas")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        for frame in frame_reader.frame_generator():
            frame_count += 1
            
            # Pipeline completo
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            active_tracks = [t for t in tracks if t.is_confirmed]
            metrics = metrics_calculator.calculate_metrics(active_tracks)
            
            # Visualizar
            rendered_frame = visualizer.render_frame(
                frame=frame,
                tracks=active_tracks,
                detections=detections,
                metrics=metrics,
                extra_info={
                    'Frame': frame_count,
                    'Tiempo': f"{time.time() - start_time:.1f}s"
                }
            )
            
            # Mostrar frame
            import cv2
            cv2.imshow('Tracking Básico', rendered_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print(f"\n📊 Estadísticas (Frame {frame_count}):")
                print(f"  Detecciones: {len(detections)}")
                print(f"  Tracks activos: {len(active_tracks)}")
                if metrics:
                    speeds = [m.speed_meters_per_second for m in metrics.values()]
                    if speeds:
                        print(f"  Velocidad promedio: {sum(speeds)/len(speeds)*3.6:.1f} km/h")
                
    finally:
        frame_reader.release()
        import cv2
        cv2.destroyAllWindows()


def ejemplo_grabacion_automatica():
    """Ejemplo con grabación automática de escenas."""
    print("🎬 Ejemplo: Grabación Automática")
    print("=" * 40)
    
    # Inicializar con recorder
    frame_reader = FrameReader(0)
    detector = YOLODetector()
    tracker = ObjectTracker()
    visualizer = Visualizer()
    recorder = Recorder("examples/output")
    
    if not frame_reader.initialize():
        print("❌ Error inicializando cámara")
        return
    
    print("📹 El sistema grabará automáticamente cuando detecte >= 2 objetos")
    print("📹 Presione 'q' para salir, 'g' para crear GIF manual")
    
    frame_count = 0
    
    try:
        for frame in frame_reader.frame_generator():
            frame_count += 1
            
            # Procesar frame
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            active_tracks = [t for t in tracks if t.is_confirmed]
            
            # Visualizar
            rendered_frame = visualizer.render_frame(
                frame=frame,
                tracks=active_tracks,
                detections=detections
            )
            
            # Auto-grabación
            recorder.auto_record_frame(rendered_frame, len(active_tracks))
            
            # Mostrar estado de grabación
            status = "🔴 GRABANDO" if recorder.is_recording else "⚪ Esperando"
            print(f"\rFrame {frame_count} | {len(active_tracks)} tracks | {status}", end="")
            
            # Mostrar frame
            import cv2
            cv2.imshow('Grabación Automática', rendered_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                gif_path = recorder.create_gif_from_buffer()
                if gif_path:
                    print(f"\n🎞️ GIF creado: {gif_path}")
                
    finally:
        frame_reader.release()
        recorder.cleanup()
        import cv2
        cv2.destroyAllWindows()
        print(f"\n✅ Procesamiento completado")


def ejemplo_configuracion_personalizada():
    """Ejemplo con configuración personalizada."""
    print("⚙️ Ejemplo: Configuración Personalizada")
    print("=" * 40)
    
    # Modificar configuración en tiempo de ejecución
    from src.utils.config_loader import config
    
    # Configurar para detectar solo personas y vehículos
    config.update('detector.classes_to_detect', [0, 1, 2, 3, 5, 7])  # person, bicycle, car, motorcycle, bus, truck
    config.update('detector.confidence_threshold', 0.7)  # Más estricto
    
    # Configurar métricas para tráfico urbano
    config.update('metrics.pixels_per_meter', 30)  # Cámara más lejana
    config.update('metrics.velocity_smoothing_window', 15)  # Más suavizado
    
    # Configurar visualización
    config.update('visualizer.show_trajectory', True)
    config.update('visualizer.trajectory_length', 50)  # Trayectorias más largas
    
    print("✅ Configuración personalizada aplicada")
    print("🎯 Detectando solo: personas y vehículos")
    print("📏 Calibración: 30 píxeles = 1 metro")
    print("🎨 Trayectorias largas habilitadas")
    
    # Ahora usar el sistema normalmente
    ejemplo_tracking_basico()


def main():
    """Función principal con menú de ejemplos."""
    print("🎯 Ejemplos del Sistema de Tracking")
    print("=" * 50)
    print("Seleccione un ejemplo:")
    print("1. Detección simple (sin tracking)")
    print("2. Tracking básico con métricas")
    print("3. Grabación automática")
    print("4. Configuración personalizada")
    print("0. Salir")
    
    while True:
        try:
            opcion = input("\n👉 Opción (0-4): ").strip()
            
            if opcion == '0':
                print("👋 ¡Hasta luego!")
                break
            elif opcion == '1':
                ejemplo_deteccion_simple()
            elif opcion == '2':
                ejemplo_tracking_basico()
            elif opcion == '3':
                ejemplo_grabacion_automatica()
            elif opcion == '4':
                ejemplo_configuracion_personalizada()
            else:
                print("❌ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrumpido por el usuario")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 