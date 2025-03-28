import requests
from PIL import Image

# Read the image file
image_path = "/rds/general/user/lw1824/home/chex/chex/dataset/MIMIC-CXR/mimic-cxr-jpg_2-0-0/files/p10/p10000032/s53189527/2a2277a9-b0ded155-c0de8eb9-c124d10e-82c5caab.jpg"
image = Image.open(image_path)
def download_sample_image() -> Image.Image:
    """Download chest X-ray with CC license."""
    base_url = "https://upload.wikimedia.org/wikipedia/commons"
    image_url = f"{base_url}/2/20/Chest_X-ray_in_influenza_and_Haemophilus_influenzae.jpg"
    headers = {"User-Agent": "RAD-DINO"}
    response = requests.get(image_url, headers=headers, stream=True)
    return Image.open(response.raw)
import torch
from transformers import AutoModel
from transformers import AutoImageProcessor
# Download the model
repo = "microsoft/rad-dino"
model = AutoModel.from_pretrained(repo)
# The processor takes a PIL image, performs resizing, center-cropping, and
# intensity normalization using stats from MIMIC-CXR, and returns a
# dictionary with a PyTorch tensor ready for the encoder
processor = AutoImageProcessor.from_pretrained(repo)
# Download and preprocess a chest X-ray

inputs = processor(images=image, return_tensors="pt")
# Encode the image!
with torch.inference_mode():
    outputs = model(**inputs)
# Look at the CLS embeddings
cls_embeddings = outputs.pooler_output
cls_embeddings.shape  # (batch_size, num_channels)
