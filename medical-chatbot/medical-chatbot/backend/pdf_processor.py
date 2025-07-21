import os
import glob
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, PodSpec

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF files"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def process_pdfs_directory(directory_path):
    """Process all PDFs in a directory and return a list of (text, metadata) tuples"""
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    documents = []
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        try:
            text = extract_text_from_pdf(pdf_file)
            if text.strip():
                documents.append({
                    "text": text,
                    "source": os.path.basename(pdf_file)
                })
                print(f"Processed: {pdf_file}")
            else:
                print(f"Warning: No text extracted from {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    return documents

def chunk_documents(documents):
    """Split documents into smaller chunks with metadata"""
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = []
    for doc in documents:
        splits = text_splitter.create_documents([doc["text"]])
        for i, split in enumerate(splits):
            chunks.append({
                "id": str(uuid.uuid4()),  # unique ID
                "text": split.page_content,
                "metadata": {
                    "source": doc["source"],
                    "chunk_index": i,
                    "text": split.page_content
                }
            })
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def initialize_pinecone():
    """Initialize Pinecone using the latest SDK"""
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    pc = Pinecone(api_key=api_key)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1024,  # Adjust based on your embedding model
            metric="cosine",
            spec=PodSpec(environment=environment)
        )
        print(f"Created new Pinecone index: {index_name}")

    index = pc.Index(index_name)
    print("Initalized pinecone")
    return index

def embed_and_upload(chunks, index):
    """Generate embeddings and upload chunks to Pinecone"""
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    texts = [chunk["text"] for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    print("Done Embedding")

    # Prepare upsert payload
    to_upsert = [
        (
            chunk["id"],     # ID
            vector,          # Values
            chunk["metadata"]  # Metadata
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    
    # Batch upload (100 per batch)
    batch_size = 100
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded {i}th batch")
    
    print(f"Uploaded {len(to_upsert)} chunks to Pinecone.")

def process_and_index_pdfs(directory_path):
    """Main function to process PDFs and index them in Pinecone"""
    print(f"Starting to process PDFs from {directory_path}")
    
    documents = process_pdfs_directory(directory_path)
    if not documents:
        print("No documents were processed. Please check the directory path and PDF files.")
        return
    
    chunks = chunk_documents(documents)
    index = initialize_pinecone()
    embed_and_upload(chunks, index)
    
    print("Indexing complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_directory = sys.argv[1]
        process_and_index_pdfs(pdf_directory)
    else:
        print("Please provide a directory path containing PDF files.")
        print("Usage: python pdf_processor.py /path/to/pdf/directory")
