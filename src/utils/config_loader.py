"""
Cargador y gestor de configuración del sistema
============================================

Este módulo maneja la carga y acceso a la configuración del sistema
desde archivos YAML.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    Cargador de configuración del sistema.
    
    Permite cargar configuraciones desde archivos YAML y proporciona
    acceso fácil a los parámetros del sistema.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el cargador de configuración.
        
        Args:
            config_path: Ruta al archivo de configuración. Si es None,
                        busca config.yaml en la carpeta config/
        """
        if config_path is None:
            # Buscar config.yaml en la carpeta config/
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Carga la configuración desde el archivo YAML."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            print(f"✅ Configuración cargada desde: {self.config_path}")
            
        except Exception as e:
            print(f"❌ Error cargando configuración: {e}")
            # Cargar configuración por defecto
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Carga una configuración mínima por defecto."""
        self._config = {
            'general': {
                'debug': False,
                'verbose': True,
                'output_dir': 'output'
            },
            'detector': {
                'model_path': 'models/yolov8n.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'tracker': {
                'max_age': 50,
                'min_hits': 3
            },
            'metrics': {
                'pixels_per_meter': 50,
                'fps': 30
            },
            'visualizer': {
                'show_trajectory': True,
                'font_scale': 0.6
            }
        }
        print("⚠️ Usando configuración por defecto")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto.
        
        Args:
            key_path: Ruta de la clave (ej: 'detector.confidence_threshold')
            default: Valor por defecto si la clave no existe
            
        Returns:
            Valor de configuración o valor por defecto
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtiene una sección completa de configuración.
        
        Args:
            section: Nombre de la sección
            
        Returns:
            Diccionario con la configuración de la sección
        """
        return self._config.get(section, {})
    
    def update(self, key_path: str, value: Any) -> None:
        """
        Actualiza un valor de configuración.
        
        Args:
            key_path: Ruta de la clave (ej: 'detector.confidence_threshold')
            value: Nuevo valor
        """
        keys = key_path.split('.')
        config_ref = self._config
        
        # Navegar hasta el penúltimo nivel
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Establecer el valor
        config_ref[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Guarda la configuración actual en un archivo YAML.
        
        Args:
            output_path: Ruta de salida. Si es None, sobreescribe el archivo original
        """
        save_path = output_path or self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as file:
                yaml.dump(self._config, file, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            print(f"✅ Configuración guardada en: {save_path}")
        except Exception as e:
            print(f"❌ Error guardando configuración: {e}")
    
    def print_config(self) -> None:
        """Imprime la configuración actual de forma legible."""
        print("\n📋 Configuración actual:")
        print("=" * 50)
        print(yaml.dump(self._config, default_flow_style=False, 
                       allow_unicode=True, indent=2))
    
    @property
    def config(self) -> Dict[str, Any]:
        """Obtiene el diccionario de configuración completo."""
        return self._config.copy()


# Instancia global del cargador de configuración
config = ConfigLoader()

# Funciones de conveniencia para acceso rápido
def get_config(key_path: str, default: Any = None) -> Any:
    """Función de conveniencia para obtener configuración."""
    return config.get(key_path, default)

def get_section(section: str) -> Dict[str, Any]:
    """Función de conveniencia para obtener una sección."""
    return config.get_section(section) 