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
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=[
        "torch",  # leave un-pinned if you support CUDA/CPU variants
        "transformers==4.51.3",
        "beautifulsoup4==4.13.4",
        "openai==1.97.0",
        "accelerate==1.9.0",
        "matplotlib",
        "rich",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
