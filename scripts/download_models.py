#!/usr/bin/env python3
"""
Script para Descargar Modelos Pre-entrenados
===========================================

Descarga los modelos YOLO necesarios para el sistema de tracking.
"""

import os
import sys
from pathlib import Path
import argparse

# Agregar src al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics no está instalado. Ejecute: pip install ultralytics")
    sys.exit(1)


def download_yolo_models(models_dir: str = "models", model_sizes: list = None):
    """
    Descarga modelos YOLO pre-entrenados.
    
    Args:
        models_dir: Directorio donde guardar los modelos
        model_sizes: Lista de tamaños de modelo a descargar
    """
    if model_sizes is None:
        model_sizes = ['n', 's', 'm']  # nano, small, medium
    
    # Crear directorio de modelos
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    print(f"📁 Creando directorio de modelos: {models_path.absolute()}")
    
    # Modelos disponibles
    available_models = {
        'n': 'yolov8n.pt',      # Nano - más rápido, menos preciso
        's': 'yolov8s.pt',      # Small - balance
        'm': 'yolov8m.pt',      # Medium - más preciso
        'l': 'yolov8l.pt',      # Large - muy preciso
        'x': 'yolov8x.pt'       # Extra Large - máxima precisión
    }
    
    print("🔄 Descargando modelos YOLO...")
    print("=" * 50)
    
    for size in model_sizes:
        if size not in available_models:
            print(f"⚠️ Tamaño '{size}' no válido. Disponibles: {list(available_models.keys())}")
            continue
        
        model_name = available_models[size]
        model_path = models_path / model_name
        
        print(f"📥 Descargando {model_name}...")
        
        try:
            # YOLO descarga automáticamente si no existe
            model = YOLO(model_name)
            
            # Mover a la carpeta de modelos si no está ahí
            if not model_path.exists():
                # Encontrar el archivo descargado
                import torch
                hub_dir = torch.hub.get_dir()
                source_path = Path(hub_dir) / "ultralytics" / model_name
                
                if source_path.exists():
                    import shutil
                    shutil.copy2(source_path, model_path)
                    print(f"✅ {model_name} copiado a {model_path}")
                else:
                    print(f"ℹ️ {model_name} descargado (ubicación por defecto)")
            else:
                print(f"✅ {model_name} ya existe")
            
            # Información del modelo
            print(f"   📊 Parámetros: ~{get_model_params(size)}")
            print(f"   ⚡ Velocidad: {get_model_speed(size)}")
            print(f"   🎯 Precisión: {get_model_accuracy(size)}")
            
        except Exception as e:
            print(f"❌ Error descargando {model_name}: {e}")
        
        print()
    
    print("✅ Descarga de modelos completada!")
    print(f"📁 Modelos guardados en: {models_path.absolute()}")


def get_model_params(size: str) -> str:
    """Obtiene información aproximada de parámetros del modelo."""
    params = {
        'n': '3.2M',
        's': '11.2M', 
        'm': '25.9M',
        'l': '43.7M',
        'x': '68.2M'
    }
    return params.get(size, 'Desconocido')


def get_model_speed(size: str) -> str:
    """Obtiene información aproximada de velocidad del modelo."""
    speeds = {
        'n': 'Muy rápido',
        's': 'Rápido',
        'm': 'Medio',
        'l': 'Lento',
        'x': 'Muy lento'
    }
    return speeds.get(size, 'Desconocido')


def get_model_accuracy(size: str) -> str:
    """Obtiene información aproximada de precisión del modelo."""
    accuracies = {
        'n': 'Básica',
        's': 'Buena',
        'm': 'Muy buena',
        'l': 'Excelente', 
        'x': 'Máxima'
    }
    return accuracies.get(size, 'Desconocido')


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Descarga modelos YOLO pre-entrenados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python download_models.py                    # Descarga modelos por defecto (n, s, m)
  python download_models.py --models n s      # Solo nano y small
  python download_models.py --models all      # Todos los modelos
  python download_models.py --dir models/yolo # Directorio personalizado
        """
    )
    
    parser.add_argument(
        '--models', 
        nargs='*', 
        default=['n', 's', 'm'],
        help='Tamaños de modelo a descargar (n, s, m, l, x) o "all" para todos'
    )
    
    parser.add_argument(
        '--dir',
        default='models',
        help='Directorio donde guardar los modelos'
    )
    
    args = parser.parse_args()
    
    # Procesar argumentos
    if 'all' in args.models:
        model_sizes = ['n', 's', 'm', 'l', 'x']
    else:
        model_sizes = args.models
    
    print("🎯 Descargador de Modelos YOLO")
    print("=" * 40)
    print(f"📁 Directorio: {args.dir}")
    print(f"🎯 Modelos: {', '.join(model_sizes)}")
    print()
    
    try:
        download_yolo_models(args.dir, model_sizes)
        print("\n🎉 ¡Descarga completada exitosamente!")
        print("\n💡 Recomendaciones:")
        print("   • yolov8n.pt: Para tiempo real en CPU")
        print("   • yolov8s.pt: Balance entre velocidad y precisión")
        print("   • yolov8m.pt: Para mayor precisión con GPU")
        
    except KeyboardInterrupt:
        print("\n⚠️ Descarga interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la descarga: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 