from utils import *
from config import *
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd



class ProductSearch:
    def __init__(self,crop_limit:int = 100):
        self.dataset = load_dataset(path_of_dataset)
        self.df = pd.DataFrame(self.dataset['train'])
        self.feature_extractor = FeatureExtractor(image_model_name=image_model_name, text_model_name=text_model_name)
        self.vector_db_handler = VectorDBHandler(image_collection_name=image_collection_name, text_collection_name=text_collection_name)
        self.text_embeddings = []
        self.image_embeddings = []
        self.ids = []
        self.prepare_data(crop_limit)
        self.add_data_to_vector_db()
    
    def prepare_data(self,crop_limit):
        self.df = self.df[:crop_limit]
        self.text_embeddings = [(self.feature_extractor.extract_text_features(self.text_to_embed(row)))[0].tolist() for _, row in self.df.iterrows()]
        self.image_embeddings = [(self.feature_extractor.extract_image_features(row['image']))[0].tolist() for _, row in self.df.iterrows()]
        self.metadatas = [self.create_metadata(row) for _,row in self.df.iterrows()]
        self.ids = [str(id) for id in self.df['id']]

    def add_data_to_vector_db(self):
        self.vector_db_handler.add_to_collection(image_embeddings=self.image_embeddings,text_embeddings=self.text_embeddings,metadatas=self.metadatas,ids=self.ids)

    def plot_results(self, results):
        num_texts = len(results['ids'][0])
        columns = min(4, num_texts) # Maximum of 4 columns
        rows = (num_texts + columns - 1) // columns

        plt.figure(figsize=(columns * 3, rows * 3))

        for index, id in enumerate(results['ids'][0]):
            image_data = np.array(self.df[self.df['id'] == float(id)]['image'])[0]

            plt.subplot(rows, columns, index + 1)
            plt.axis('off')
            plt.imshow(image_data)
            plt.title(results['metadatas'][0][index]['product_name'], fontsize=7)

        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()

    def run(self,query :str,n_results: int = n_results):
        query_type = False
        if isinstance(query,str):    
            query_embeddings = self.feature_extractor.extract_text_features(query)
            query_type = True
        else:
            query_embeddings = self.feature_extractor.extract_image_features(query)
        results = self.vector_db_handler.retrieve_products(query_embeddings=query_embeddings,n_results=n_results,query_type=query_type)
        self.plot_results(results)

    def text_to_embed(self,row):
        return f"{row['productDisplayName']} {row['articleType']} {row['baseColour']} {row['usage']} {row['gender']}"

    def create_metadata(self,row):
        metadata = {'id':row['id'],
                'gender':row['gender'],
                'masterCategory':row['masterCategory'],
                'subCategory':row['subCategory'],
                'articleType':row['articleType'],
                'baseColour':row['baseColour'],
                'season':row['season'],
                'usage':row['usage'],
                'product_name':row['productDisplayName']
                }
        return metadata

    def see_head(self):
        return self.df.head()


if __name__=='__main__':
    app = ProductSearch(crop_limit=10)
    app.run(query='puma tshirts for men',n_results=5)
    # app.run(query = app.df['image'][0],n_results=4)
