import torch
import requests
from PIL import Image
from transformers import AutoProcessor, SiglipModel

class RewardEstimator:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.model = SiglipModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def get_embeddings(self, image: Image.Image, text: str):
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.image_embeds, outputs.text_embeds, outputs.logits_per_image

    
    
if __name__ == "__main__":
    estimator = RewardEstimator()
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "A person riding a horse on a beach"
    image_embeds, text_embeds, logits_per_image = estimator.get_embeddings(image, text)
    print("Image Embeddings:", image_embeds.shape) # (1, 768)
    print("Text Embeddings:", text_embeds.shape) # (1, 768)
    print("Logits per Image:", logits_per_image.shape) # (1, 1) this is the image-text similarity score