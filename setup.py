#!/usr/bin/env python3
"""
Setup script para el Sistema de Detección y Seguimiento de Objetos
================================================================

Script de instalación que permite instalar el proyecto como un paquete
de Python para facilitar su distribución y uso.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README para la descripción larga
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Leer requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="object-tracking-system",
    version="1.0.0",
    author="Parcial Visual Computing",
    author_email="student@university.edu",
    description="Sistema completo de detección y seguimiento de objetos en tiempo real",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/parcial-visual",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": [
            "cupy-cuda11x>=11.0.0",  # Para aceleración GPU
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
    },
    entry_points={
        "console_scripts": [
            "object-tracking=main:main",
            "object-tracking-gui=gui_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md", "*.txt"],
        "config": ["*.yaml"],
    },
    keywords=[
        "computer vision",
        "object detection",
        "object tracking",
        "yolo",
        "deepsort",
        "opencv",
        "real-time",
        "video analysis",
        "artificial intelligence",
        "machine learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/username/parcial-visual/issues",
        "Source": "https://github.com/username/parcial-visual",
        "Documentation": "https://github.com/username/parcial-visual/blob/main/README.md",
    },
) 