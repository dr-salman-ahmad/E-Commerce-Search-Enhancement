#! pip install timm ftfy regex tqdm datasets
#! pip install git+https://github.com/openai/CLIP.git
import os
import clip
import torch
import skimage
import textwrap
import requests
import numpy as np
import pandas as pd
from PIL import Image
import IPython.display
import matplotlib.pyplot as plt
from datasets import load_dataset
from pkg_resources import packaging

class CLIPModelEvaluator:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model = self.model.eval().cuda() if self.device == "cuda" else self.model.eval()
        self.text_wrapper = textwrap.TextWrapper(width=30)
        
    def preprocess_images_and_texts(self, df, sample_size=100):
        random_sample = df.sample(n=sample_size, random_state=2)
        image_data = random_sample['image'].tolist()
        descriptions = random_sample['productDisplayName'].tolist()
        
        images, texts = [], []
        for index, image in enumerate(image_data):
            try:
                wrapped_text = self.text_wrapper.fill(text=descriptions[index])
                images.append(self.preprocess(image))
                texts.append(descriptions[index])
            except Exception as e:
                print(f"Error processing image or text at index {index}: {e}")
        
        return images, texts

    def compute_similarity(self, images, texts):
        try:
            image_input = torch.tensor(np.stack(images)).cuda() if self.device == "cuda" else torch.tensor(np.stack(images))
            text_tokens = clip.tokenize([desc for desc in texts]).cuda() if self.device == "cuda" else clip.tokenize([desc for desc in texts])

            with torch.no_grad():
                image_features = self.model.encode_image(image_input).float()
                text_features = self.model.encode_text(text_tokens).float()
                
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

            return similarity
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return None

    def compute_accuracy(self, similarity, k, count):
        try:
            top_k_indices = np.argsort(similarity, axis=1)[:, -k:]
            correct_indices = np.arange(len(similarity)).reshape(-1, 1)
            matches_top_k = np.any(top_k_indices == correct_indices, axis=1)
            top_k_accuracy = matches_top_k.mean()
            print(f"Evaluating {count} images for Top-{k} Accuracy of matching descriptions to images: {top_k_accuracy:.2f}")
        except Exception as e:
            print(f"Error computing accuracy for Top-{k}: {e}")

def load_dataset_sample():
    try:
        ds = load_dataset("ashraq/fashion-product-images-small")
        df = pd.DataFrame(ds['train'].select(range(1000)))
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def main():
    evaluator = CLIPModelEvaluator()
    
    df = load_dataset_sample()
    if df is not None:
        images, texts = evaluator.preprocess_images_and_texts(df, sample_size=100)
        if images and texts:
            similarity = evaluator.compute_similarity(images, texts)
            if similarity is not None:
                count = len(texts)
                for k in [1, 3, 5, 10, 20, 30]:
                    evaluator.compute_accuracy(similarity, k, count)

if __name__ == "__main__":
    main()
