from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTConfig

from PIL import Image
import cv2
import torch
import time

model_name = "google/owlvit-base-patch32"
processor = OwlViTProcessor.from_pretrained(model_name)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = OwlViTForObjectDetection.from_pretrained(model_name, config=OwlViTConfig(return_dict=False)).to(device)

classes = ["no object", "orange mallet or hammer", "water bottle"]
example = processor(text=classes, return_tensors="pt")
example = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in example.items()}

print(example["input_ids"])
print()
print(example["attention_mask"])
