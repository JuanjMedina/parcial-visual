#!/usr/bin/env python3
"""
Interfaz Gráfica del Sistema de Tracking
========================================

Interfaz gráfica usando PySimpleGUI para controlar
el sistema de detección y seguimiento de objetos.
"""

import PySimpleGUI as sg
import cv2
import numpy as np
import threading
import time
from pathlib import Path
import io
from PIL import Image

from src.core.frame_reader import FrameReader
from src.core.detector import YOLODetector
from src.core.tracker import ObjectTracker
from src.core.metrics_calculator import MetricsCalculator
from src.core.visualizer import Visualizer
from src.core.recorder import Recorder
from src.utils.config_loader import get_config


class TrackingGUI:
    """
    Interfaz gráfica para el sistema de tracking.
    
    Proporciona controles intuitivos y visualización en tiempo real
    del sistema de detección y seguimiento.
    """
    
    def __init__(self):
        """Inicializa la interfaz gráfica."""
        # Configurar tema
        sg.theme(get_config('gui.theme', 'DarkBlue3'))
        
        # Configuración de ventana
        self.window_size = get_config('gui.window_size', [1200, 800])
        self.display_size = (640, 480)
        
        # Sistema de tracking
        self.system_initialized = False
        self.is_running = False
        self.is_paused = False
        
        # Componentes del sistema
        self.frame_reader = None
        self.detector = None
        self.tracker = None
        self.metrics_calculator = None
        self.visualizer = None
        self.recorder = None
        
        # Threading
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # Estadísticas
        self.stats = {
            'fps': 0,
            'frame_count': 0,
            'detections': 0,
            'tracks': 0,
            'elapsed_time': 0
        }
        
        # Crear layout y ventana
        self.window = self._create_window()
    
    def _create_window(self) -> sg.Window:
        """Crea la ventana principal."""
        # Panel de control izquierdo
        control_panel = [
            [sg.Text('🎯 Sistema de Tracking', font=('Arial', 16, 'bold'))],
            [sg.HSeparator()],
            
            # Fuente de video
            [sg.Text('📹 Fuente de Video:')],
            [sg.Radio('Cámara', 'source', key='camera', default=True),
             sg.Spin([i for i in range(5)], initial_value=0, key='camera_id', size=(3, 1))],
            [sg.Radio('Archivo', 'source', key='file'),
             sg.Input(key='video_path', size=(25, 1)),
             sg.FileBrowse(file_types=(("Videos", "*.mp4 *.avi *.mov"),))],
            
            [sg.HSeparator()],
            
            # Configuración del detector
            [sg.Text('🎯 Configuración YOLO:')],
            [sg.Text('Confianza:'), sg.Slider(range=(0.1, 1.0), default_value=0.5, 
                                             resolution=0.1, orientation='h', key='confidence')],
            [sg.Text('Dispositivo:'), sg.Combo(['cpu', 'cuda'], default_value='cpu', key='device')],
            
            [sg.HSeparator()],
            
            # Configuración de métricas
            [sg.Text('📏 Calibración:')],
            [sg.Text('Píxeles/metro:'), sg.Input('50', key='pixels_per_meter', size=(10, 1))],
            [sg.Text('FPS video:'), sg.Input('30', key='video_fps', size=(10, 1))],
            
            [sg.HSeparator()],
            
            # Controles de visualización
            [sg.Text('🎨 Visualización:')],
            [sg.Checkbox('Mostrar trayectorias', default=True, key='show_trajectories')],
            [sg.Checkbox('Mostrar métricas', default=True, key='show_metrics')],
            [sg.Checkbox('Mostrar FPS', default=True, key='show_fps')],
            
            [sg.HSeparator()],
            
            # Controles principales
            [sg.Button('🚀 Inicializar', key='init', size=(12, 1))],
            [sg.Button('▶️ Iniciar', key='start', disabled=True, size=(12, 1))],
            [sg.Button('⏸️ Pausar', key='pause', disabled=True, size=(12, 1))],
            [sg.Button('⏹️ Detener', key='stop', disabled=True, size=(12, 1))],
            
            [sg.HSeparator()],
            
            # Controles de grabación
            [sg.Text('🎬 Grabación:')],
            [sg.Button('🔴 Grabar Video', key='record_video', size=(12, 1))],
            [sg.Button('🎞️ Crear GIF', key='create_gif', size=(12, 1))],
            
            [sg.HSeparator()],
            
            # Estadísticas
            [sg.Text('📊 Estadísticas:', font=('Arial', 12, 'bold'))],
            [sg.Text('FPS: 0', key='fps_display')],
            [sg.Text('Frames: 0', key='frames_display')],
            [sg.Text('Detecciones: 0', key='detections_display')],
            [sg.Text('Tracks: 0', key='tracks_display')],
            [sg.Text('Tiempo: 0s', key='time_display')],
            
            [sg.HSeparator()],
            [sg.Button('❌ Salir', key='exit')]
        ]
        
        # Panel de video principal
        video_panel = [
            [sg.Text('📺 Video en Tiempo Real', font=('Arial', 14, 'bold'))],
            [sg.Image(key='video_display', size=self.display_size)],
            [sg.Text('Estado: Esperando...', key='status', font=('Arial', 10))]
        ]
        
        # Layout principal
        layout = [
            [sg.Column(control_panel, vertical_alignment='top', size=(300, 700), scrollable=True),
             sg.VSeperator(),
             sg.Column(video_panel, element_justification='center')]
        ]
        
        return sg.Window(
            get_config('gui.window_title', 'Sistema de Tracking'),
            layout,
            size=self.window_size,
            finalize=True,
            resizable=True
        )
    
    def run(self):
        """Ejecuta la interfaz gráfica."""
        print("🎮 Iniciando interfaz gráfica...")
        
        try:
            while True:
                event, values = self.window.read(timeout=100)
                
                if event in (sg.WIN_CLOSED, 'exit'):
                    break
                
                # Manejar eventos
                if event == 'init':
                    self._initialize_system(values)
                elif event == 'start':
                    self._start_processing()
                elif event == 'pause':
                    self._pause_resume()
                elif event == 'stop':
                    self._stop_processing()
                elif event == 'record_video':
                    self._toggle_recording()
                elif event == 'create_gif':
                    self._create_gif()
                
                # Actualizar estadísticas si el sistema está corriendo
                if self.is_running:
                    self._update_display()
        
        except Exception as e:
            sg.popup_error(f"Error en la interfaz: {e}")
        
        finally:
            self._cleanup()
    
    def _initialize_system(self, values):
        """Inicializa el sistema con los valores de configuración."""
        try:
            self.window['status'].update('🔄 Inicializando sistema...')
            self.window.refresh()
            
            # Determinar fuente de video
            if values['camera']:
                source = int(values['camera_id'])
            else:
                source = values['video_path'] if values['video_path'] else None
            
            if not source and not values['camera']:
                sg.popup_error("Por favor seleccione una fuente de video")
                return
            
            # Inicializar componentes
            self.frame_reader = FrameReader(source)
            self.detector = YOLODetector()
            self.tracker = ObjectTracker()
            self.metrics_calculator = MetricsCalculator()
            self.visualizer = Visualizer()
            self.recorder = Recorder()
            
            # Actualizar configuración
            self.detector.update_threshold(values['confidence'])
            self.metrics_calculator.update_calibration(
                float(values['pixels_per_meter']),
                float(values['video_fps'])
            )
            
            # Configurar visualizador
            self.visualizer.show_trajectory = values['show_trajectories']
            self.visualizer.show_fps = values['show_fps']
            
            # Inicializar frame reader
            if not self.frame_reader.initialize():
                sg.popup_error("Error inicializando fuente de video")
                return
            
            self.system_initialized = True
            
            # Actualizar botones
            self.window['init'].update(disabled=True)
            self.window['start'].update(disabled=False)
            self.window['status'].update('✅ Sistema inicializado')
            
            sg.popup('✅ Sistema inicializado correctamente!', title='Éxito')
            
        except Exception as e:
            sg.popup_error(f"Error inicializando sistema: {e}")
            self.window['status'].update('❌ Error en inicialización')
    
    def _start_processing(self):
        """Inicia el procesamiento en tiempo real."""
        if not self.system_initialized:
            sg.popup_error("Primero inicialice el sistema")
            return
        
        self.is_running = True
        self.is_paused = False
        self.stop_processing.clear()
        
        # Iniciar hilo de procesamiento
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Actualizar botones
        self.window['start'].update(disabled=True)
        self.window['pause'].update(disabled=False)
        self.window['stop'].update(disabled=False)
        self.window['status'].update('▶️ Procesando...')
    
    def _pause_resume(self):
        """Pausa o reanuda el procesamiento."""
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.window['pause'].update('▶️ Reanudar')
            self.window['status'].update('⏸️ Pausado')
        else:
            self.window['pause'].update('⏸️ Pausar')
            self.window['status'].update('▶️ Procesando...')
    
    def _stop_processing(self):
        """Detiene el procesamiento."""
        self.is_running = False
        self.stop_processing.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        # Actualizar botones
        self.window['start'].update(disabled=False)
        self.window['pause'].update(disabled=True)
        self.window['stop'].update(disabled=True)
        self.window['pause'].update('⏸️ Pausar')
        self.window['status'].update('⏹️ Detenido')
    
    def _toggle_recording(self):
        """Alterna la grabación de video."""
        if not self.recorder:
            return
        
        if not self.recorder.is_recording:
            timestamp = time.strftime("%H%M%S")
            if self.recorder.start_video_recording(f"gui_recording_{timestamp}.mp4"):
                self.window['record_video'].update('⏹️ Detener Grabación')
                sg.popup_quick_message('🔴 Grabación iniciada', auto_close_duration=2)
        else:
            video_path = self.recorder.stop_video_recording()
            self.window['record_video'].update('🔴 Grabar Video')
            if video_path:
                sg.popup_quick_message(f'💾 Video guardado: {Path(video_path).name}', 
                                     auto_close_duration=3)
    
    def _create_gif(self):
        """Crea un GIF del buffer actual."""
        if not self.recorder:
            return
        
        timestamp = time.strftime("%H%M%S")
        gif_path = self.recorder.create_gif_from_buffer(f"gui_gif_{timestamp}.gif")
        
        if gif_path:
            sg.popup_quick_message(f'🎞️ GIF creado: {Path(gif_path).name}', 
                                 auto_close_duration=3)
        else:
            sg.popup_error("Error creando GIF o buffer vacío")
    
    def _processing_loop(self):
        """Loop principal de procesamiento."""
        start_time = time.time()
        frame_count = 0
        
        try:
            for frame in self.frame_reader.frame_generator():
                if self.stop_processing.is_set():
                    break
                
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Procesar frame
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)
                active_tracks = [t for t in tracks if t.is_confirmed]
                metrics = self.metrics_calculator.calculate_metrics(active_tracks)
                
                # Renderizar
                processed_frame = self.visualizer.render_frame(
                    frame=frame,
                    tracks=active_tracks,
                    detections=detections,
                    metrics=metrics
                )
                
                # Grabar si está habilitado
                if self.recorder.is_recording:
                    self.recorder.record_frame(processed_frame)
                
                # Auto-grabación
                self.recorder.auto_record_frame(processed_frame, len(active_tracks))
                
                # Actualizar estadísticas
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                self.stats.update({
                    'fps': fps,
                    'frame_count': frame_count,
                    'detections': len(detections),
                    'tracks': len(active_tracks),
                    'elapsed_time': elapsed_time
                })
                
                # Actualizar display de video
                self._update_video_display(processed_frame)
                
        except Exception as e:
            print(f"Error en procesamiento: {e}")
        
        finally:
            self.is_running = False
    
    def _update_video_display(self, frame: np.ndarray):
        """Actualiza la visualización del video."""
        try:
            # Redimensionar frame para display
            display_frame = cv2.resize(frame, self.display_size)
            
            # Convertir a formato para PySimpleGUI
            img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Convertir a bytes
            bio = io.BytesIO()
            img_pil.save(bio, format='PNG')
            img_bytes = bio.getvalue()
            
            # Actualizar display
            self.window['video_display'].update(data=img_bytes)
            
        except Exception as e:
            print(f"Error actualizando display: {e}")
    
    def _update_display(self):
        """Actualiza las estadísticas en pantalla."""
        try:
            self.window['fps_display'].update(f"FPS: {self.stats['fps']:.1f}")
            self.window['frames_display'].update(f"Frames: {self.stats['frame_count']}")
            self.window['detections_display'].update(f"Detecciones: {self.stats['detections']}")
            self.window['tracks_display'].update(f"Tracks: {self.stats['tracks']}")
            self.window['time_display'].update(f"Tiempo: {self.stats['elapsed_time']:.1f}s")
        except:
            pass
    
    def _cleanup(self):
        """Limpia recursos."""
        print("🧹 Limpiando interfaz gráfica...")
        
        # Detener procesamiento
        if self.is_running:
            self._stop_processing()
        
        # Liberar recursos
        if self.frame_reader:
            self.frame_reader.release()
        
        if self.recorder and self.recorder.is_recording:
            self.recorder.stop_video_recording()
        
        if self.recorder:
            self.recorder.cleanup()
        
        # Cerrar ventana
        self.window.close()


def main():
    """Función principal de la GUI."""
    try:
        # Verificar disponibilidad de PySimpleGUI
        import PySimpleGUI as sg
        
        # Crear y ejecutar interfaz
        gui = TrackingGUI()
        gui.run()
        
    except ImportError:
        print("❌ PySimpleGUI no está instalado")
        print("📦 Instale con: pip install PySimpleGUI")
    except Exception as e:
        print(f"❌ Error en la interfaz gráfica: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 