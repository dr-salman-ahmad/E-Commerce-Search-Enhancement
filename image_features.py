#!pip install datasets sentence-transformers chromadb
# Import necessary libraries
import requests
import numpy as np
import pandas as pd
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import chromadb
import matplotlib.pyplot as plt

class ImageFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def extract_image_features(self, image):
        image_inputs = self.processor(images=image, return_tensors="pt", padding=True)
        image_features = self.model.get_image_features(**image_inputs).detach().numpy()
        image_features = image_features / np.linalg.norm(image_features)
        return image_features

class ChromaDBManager:
    def __init__(self, collection_name="products"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_to_collection(self, embeddings, metadatas, ids):
        self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    def query_collection(self, query_embeddings, n_results=6):
        return self.collection.query(query_embeddings=query_embeddings, n_results=n_results)

class ImageSearchApp:
    def __init__(self, dataset_name="ashraq/fashion-product-images-small"):
        self.dataset = load_dataset(dataset_name)
        self.df = pd.DataFrame(self.dataset['train'])
        self.feature_extractor = ImageFeatureExtractor()
        self.db_manager = ChromaDBManager()

    def prepare_data(self, limit=100):
        self.df = self.df[:limit]
        self.embeddings = [self.feature_extractor.extract_image_features(row['image'])[0].tolist() for _, row in self.df.iterrows()]
        self.metadatas = [{'product_name': product} for product in self.df['productDisplayName']]
        self.ids = [str(i) for i in self.df['id']]
        self.db_manager.add_to_collection(self.embeddings, self.metadatas, self.ids)

    def plot_results(self, results):
        num_images = len(results['ids'][0])
        max_columns = 3
        columns = min(max_columns, num_images)
        rows = (num_images + columns - 1) // columns

        plt.figure(figsize=(columns * 3, rows * 3))

        for index, image_id in enumerate(results['ids'][0]):
            image_data = np.array(self.df[self.df['id'] == float(image_id)]['image'])[0]

            plt.subplot(rows, columns, index + 1)
            plt.axis('off')
            plt.imshow(image_data)
            plt.title(results['metadatas'][0][index]['product_name'], fontsize=7)

        plt.tight_layout()
        plt.show()

    def run(self):
        # Prepare data and extract features
        self.prepare_data()

        # Test with the first image in the DataFrame
        test_image_features = self.feature_extractor.extract_image_features(self.df.iloc[0]['image']).tolist()

        # Query the collection for similar products
        search_results = self.db_manager.query_collection(query_embeddings=test_image_features, n_results=6)

        # Plot the results
        self.plot_results(search_results)

def main():
    # Initialize the app
    app = ImageSearchApp()
    # Run the app
    app.run()

if __name__ == "__main__":
    main()
