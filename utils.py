from transformers import AutoModel
import numpy as np
import chromadb
from transformers import CLIPProcessor, CLIPModel
from config import *


class VectorDBHandler:
    def __init__(self):
        self.client = chromadb.Client()  # Initialize ChromaDB client
        self.image_collection = self.create_new_collection(image_collection_name)
        self.text_collection = self.create_new_collection(text_collection_name)

    def create_new_collection(self, collection_name):
        try:
            return self.client.get_or_create_collection(name=collection_name)  # Create or retrieve the collection
        except Exception as e:
            print(f"Error creating or retrieving collection '{collection_name}'. Error: {e}")
            return None

    def add_to_collection(self, image_embeddings, text_embeddings, metadatas, ids):
        try:
            self.text_collection.add(embeddings=text_embeddings, metadatas=metadatas, ids=ids)
            self.image_collection.add(embeddings=image_embeddings, metadatas=metadatas, ids=ids)
            print("Data added to collection successfully.")
        except Exception as e:
            print(f"Error adding data to collection . Error: {e}")

    def retrieve_products(self, query_embeddings, n_results, query_type):
        try:
            if query_type:
                return self.text_collection.query(query_embeddings=query_embeddings, n_results=n_results)
            else:
                return self.image_collection.query(query_embeddings=query_embeddings, n_results=n_results)
        except Exception as e:
            print(f"Error while retrieving products. Error: {e}")
            return None


class FeatureExtractor:
    def __init__(self, image_model_name, text_model_name):
        try:
            self.text_model = AutoModel.from_pretrained(text_model_name,
                                                        trust_remote_code=True)  # trust_remote_code is needed to use the encode method
            self.image_model = CLIPModel.from_pretrained(image_model_name)
            self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        except Exception as e:
            print(f"Failed to load models. Error: {e}")

    def extract_text_features(self, text):
        try:
            text_features = self.text_model.encode(text).reshape(1, -1)
            text_features = text_features / np.linalg.norm(text_features)
            return text_features
        except Exception as e:
            print(f"Error during feature extraction for text '{text}'. Error: {e}")
            return None

    def extract_image_features(self, image):
        try:
            image_inputs = self.image_processor(images=image, return_tensors="pt", padding=True)
            image_features = self.image_model.get_image_features(**image_inputs).detach().numpy()
            image_features = image_features / np.linalg.norm(image_features)
            return image_features
        except Exception as e:
            print(f"Error during feature extraction of image. Error: {e}")
            return None
