import torch
import json
import os
import chromadb

from tqdm.auto import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

if torch.backends.mps.is_available():
    print(f"GPU available: MPS")
    DEVICE = "mps"
else:
    print("GPU isn't available, using CPU")



embeddings = HuggingFaceBgeEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE}
)

def data_processing(data_dir):
    print("Starting to read Json Files:")
    docs = list()
    file_count = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    for filename in tqdm(os.listdir(data_dir), desc="Loading files"):
        if filename.endswith(".json"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                text = json.dumps(data, indent=2)
                docs.append(text)
    
    print("Starting the text splitting....")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.create_documents(docs)
    total_chunks = len(splits)
    print(f"Created {total_chunks} text chunks")

    print("Creating ChromaDB database...")
    chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="chroma_db"
        ))
    
    batch_size = 64
    total_batches = (total_chunks + batch_size - 1)
    print(f"Processing embeddings in {total_batches}")
    with tqdm(total=total_chunks, desc="Creating embeddings...") as progbar:
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]

            if i == 0:
                vector_store = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    client=chroma_client,
                    collection_name="pharma-kb"
                )
            else:
                vector_store.add_documents(documents=batch)
                progbar.update(len(batch))
                if DEVICE == "cuda" and i % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
    print("ChromaDB database created successfully")

def main():
    data_dir = "../datasets/microlabs_usa/" # folder location of the json files
    data_processing(data_dir)

if __name__ == "__main__":
    main()