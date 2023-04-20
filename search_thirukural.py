import streamlit as st
import pandas as pd
from typing import List
import hnswlib
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder


class HnswIndex:

    def __init__(self, embedder: SentenceTransformer, dimension: int = 768):
        self.embedder = embedder
        self.dimension = dimension
        self.position = 0
        self.documents = []
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.index.init_index(max_elements=1500, ef_construction=200, M=16)

    def add_to_index(self, sentences: List[str]) -> None:
        self.documents.extend(sentences)
        embeddings = self.embedder.encode(sentences)
        self.index.add_items(embeddings)
        print(f'There are now a total of {self.index.element_count} vectors.')

    def search(self, query_text: str, k: int = 3):
        query = self.embedder.encode(query_text)
        self.index.set_ef(25)
        ann, distances = self.index.knn_query(query, k)
        # distances, ann = self.index.search(query, k)

        # results = pd.DataFrame(columns=['Index', 'Score', 'Sentence'])
        #
        # for idx, score in zip(ann[0], distances[0]):
        #     row = [idx, score, self.documents[idx].strip()]
        #     results.loc[len(results)] = row
        return ann[0]

@st.cache_data
def load_kurals():
    df = pd.read_excel(r'tamil_kurals.xlsx')
    return df

@st.cache_data
def load_ann_searcher(sentences, embedder_model_name, dimension):
    embedder_model = SentenceTransformer(embedder_model_name)
    ann_searcher = HnswIndex(embedder_model, dimension=dimension)
    ann_searcher.add_to_index(sentences)
    return ann_searcher

@st.cache_data
def load_crossencoder():
    crossencoder_model_name = 'cross-encoder/ms-marco-MiniLM-L-4-v2'
    crossencoder_model = CrossEncoder(crossencoder_model_name, max_length=512, num_labels=1)
    return crossencoder_model

kurals = load_kurals()
explanation_sentences = kurals.iloc[:,2]
embedder_model_name = "sentence-transformers/all-MiniLM-L12-v2"
dimension = 384
ann_searcher = load_ann_searcher(explanation_sentences, embedder_model_name, dimension)
crossencoder_model = load_crossencoder()

#print(sentences)
#embedder_model_name = 'msmarco-distilbert-base-v4'  # A good balance between speed and accuracy.
#pd.set_option('display.max_colwidth', None)

st.title('Ask Valluvar')
st.caption("Semantic Search with Transformers on Thirukkural")
st.divider()
with st.spinner('Transformer is searching...'):
    query_text = st.text_input('Describe your topic of interest in a few words', '').strip()
    nearest_neighbor_indices = ann_searcher.search(query_text, k=40)
    nearest_neighbors=kurals.loc[nearest_neighbor_indices]
    subset=[]
    for r in nearest_neighbors.iloc[:,2]:
        subset.append([query_text, r])
    crossencoder_score = crossencoder_model.predict(subset, convert_to_tensor=True)
    nearest_neighbors["Score"] = torch.nn.Sigmoid()(crossencoder_score)
    results = nearest_neighbors.sort_values(by="Score", ascending=False).iloc[:10]


for index, row in results.iterrows():
    st.markdown("**"+"Kural "+ str(row["Kural Number"])+"**")
    st.markdown("**"+row["Kural In Tamil"]+"**")
    st.markdown(row["Explanation"])
    st.divider()