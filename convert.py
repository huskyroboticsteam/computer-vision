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

classes = ["hammer", "sunglasses", "a solid orange mallet or hammer", "headphones"]
example = processor(text=classes, images=Image.open("mallet.jpg"), return_tensors="pt")
example = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in example.items()}
# example["return_dict"] = True
print(type(example["input_ids"]))
print(type(example["pixel_values"]))
print(type(example["attention_mask"]))
traced_script_module = torch.jit.trace(detector, example_kwarg_inputs=example)

traced_script_module.save("owlvit-cpp.pt")