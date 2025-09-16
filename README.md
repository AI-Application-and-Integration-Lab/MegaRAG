# MegaRAG: Multimodal Graph-based Retrieval Augmented Generation

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-custom-lightgrey">
</p>

**MegaRAG** enables **global visual question answering** on documents by constructing a **Multimodal Knowledge Graph (MMKG)**. It combines graph-based reasoning with page retrieval techniques for precise and rich responses.


## 🚀 Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/219ae758-b54c-45c9-a261-b63644ffec08" style="width:90%;" alt="MegaRAG Architecture" />
</p>

## 📦 Installation
#### Requirements

* Python 3.9+ (3.10 recommended)
* GPU for embeddings inference
* OpenAI API key

### Step 1: Install [MinerU](https://github.com/opendatalab/MinerU/tree/release-1.3.6)

```bash
git clone -b release-1.3.6 https://github.com/opendatalab/MinerU.git
cd MinerU

conda create --name mineru python=3.10 -y
conda activate mineru
pip install -e .

pip install huggingface_hub
wget https://raw.githubusercontent.com/opendatalab/MinerU/refs/heads/release-1.3.6/scripts/download_models_hf.py -O download_models_hf.py

sed -i "s|https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json|https://raw.githubusercontent.com/opendatalab/MinerU/release-1.3.6/magic-pdf.template.json|" download_models_hf.py

python download_models_hf.py
```

### Step 2: Install MegaRAG

```bash
git clone https://github.com/AI-Application-and-Integration-Lab/MegaRAG.git
cd MegaRAG

# Install missing packages
conda activate mineru
pip install -r requirements_mineru.txt

conda create --name megarag python=3.10 -y
conda activate megarag
pip install -e .

# Fill in your OPENAI_API_KEY and MINERU_PATH in env.sh
cp .env.sh env.sh

# Install lightRAG
mkdir lib && cd lib
git clone --branch v1.4.3 https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .
```

## ⚡ Quickstart

### 1. Use the Tiny Example

```bash
cd egs/world_history_tiny
mkdir data
```

### 2. Download Example PDF

Download the example PDF, and query file from [Google Drive](https://drive.google.com/drive/folders/1iuukUWsxMYobuDRLRJ3dBOkB9mdPGoPp?usp=sharing) and place it in the `data/` folder.

### 3. Build the Multimodal Knowledge Graph

```bash
bash ./run_build_mmkg.sh
```

### 4. Query with MegaRAG

```bash
bash ./run_querying.sh
```

## 📂 Using Your Own Dataset

### 1. Create a New Recipe

```bash
cp -r egs/.template egs/<your_dataset>
# Change <your_dataset> to your dataset name
cd egs/<your_dataset>
mkdir data
```

### 2. Add Your Data

Place your documents (PDF) in the `data/` folder.

### 3. Edit the Config File

Modify the entity types or other settings in `conf/addon_params.yaml` to match your data.

### 4. Build and Query

```bash
bash egs/<your_dataset>/run_build_mmkg.sh
bash egs/<your_dataset>/run_querying.sh
# Make sure to update filenames in these scripts accordingly.
```

## Acknowledgments

MegaRAG is inspired by the work of [LightRAG](https://github.com/HKUDS/LightRAG). We are grateful for their excellent tools and contributions.

## 📄 License

This project is released under a custom license.
See the [LICENSE](./LICENSE) file for full terms and conditions.

For academic or commercial use, please contact the authors directly.
