import os
import torch
import numpy as np
import pipmaster as pm  # Pipmaster for dynamic library install
import torch.nn.functional as F

# install specific modules
if not pm.is_installed("transformers"):
    pm.install("transformers")
if not pm.is_installed("torch"):
    pm.install("torch")
if not pm.is_installed("numpy"):
    pm.install("numpy")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def hf_embed(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    # Detect the appropriate device
    if torch.cuda.is_available():
        device = next(embed_model.parameters()).device  # Use CUDA if available
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS for Apple Silicon
    else:
        device = torch.device("cpu")  # Fallback to CPU

    # Move the model to the detected device
    embed_model = embed_model.to(device)

    # Tokenize the input texts and move them to the same device
    encoded_texts = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    # Perform inference
    with torch.no_grad():
        outputs = embed_model(
            input_ids=encoded_texts["input_ids"],
            attention_mask=encoded_texts["attention_mask"],
        )
        embeddings = outputs.last_hidden_state.mean(dim=1)

    # Convert embeddings to NumPy
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()

async def hf_gme_embed(embed_model, texts: list[str]=[], images: list[str]=[], is_query: bool=False):
    instruction = "Find an image that matches the given text."
    if len(images) == 0:
        embeddings = embed_model.get_text_embeddings(
            texts=texts, 
            instruction=instruction if is_query else "", 
            is_query=is_query
        )
    elif len(texts) == 0:
        embeddings = embed_model.get_image_embeddings(
            images=images, 
            instruction=instruction if is_query else "", 
            is_query=is_query
        )
    else:
        embeddings = embed_model.get_fused_embeddings(
            texts=texts, 
            images=images, 
            instruction=instruction if is_query else "", 
            is_query=is_query
        )
    # Convert embeddings to NumPy
    if embeddings.dtype == torch.bfloat16:
        return embeddings.detach().to(torch.float32).cpu().numpy()
    else:
        return embeddings.detach().cpu().numpy()
