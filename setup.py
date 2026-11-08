from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

setup(
    name="megarag",
    version="0.1.0",
    description="MegaRAG: Multimodal Graph-based RAG",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=[
        "lightrag-hku==1.4.3",
        "nano-vectordb==0.0.4.3",
        "networkx>=3.0,<4",
        "numpy>=1.24,<3",
        "pandas>=2,<3",
        "PyYAML>=6,<7",
        "Pillow>=10,<12",
        "tenacity>=8,<10",
        "beautifulsoup4==4.13.4",
        "openai==1.97.0",
        "matplotlib",
        "rich",
    ],
    extras_require={
        "local": ["torch>=2.2", "transformers==4.51.3", "accelerate==1.9.0"],
        "pdf": ["PyMuPDF==1.24.14"],
        "dev": ["pytest>=8,<10", "build>=1,<2"],
    },
    entry_points={"console_scripts": ["megarag=megarag.cli:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
