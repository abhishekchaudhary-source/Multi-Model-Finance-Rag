import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load configuration from .env file
load_dotenv()

DATA_DIR = "Data"
COMPANIES = ["Amazon", "Apple", "Google", "Meta"]

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Fintech Collection")


def extract_metadata_from_filename(filename, company):
    """
    Extracts structured metadata like doc_type, quarter, and year from filename.
    e.g. 'amazon 10-q q2 2024.pdf' -> {'doc_type': '10-Q', 'quarter': 'Q2', 'year': '2024'}
    """
    metadata = {
        "company": company,
        "filename": filename
    }
    
    # Extract year (4 digits)
    year_match = re.search(r'(20\d\d)', filename)
    if year_match:
        metadata["year"] = year_match.group(1)
        
    # Extract document type (10-K, 10-Q, 8-K)
    doc_type_match = re.search(r'(10-k|10-q|8-k)', filename, re.IGNORECASE)
    if doc_type_match:
        metadata["doc_type"] = doc_type_match.group(1).upper()
        
    # Extract quarter if present (q1, q2, q3, q4)
    quarter_match = re.search(r'\b(q[1-4])\b', filename, re.IGNORECASE)
    if quarter_match:
        metadata["quarter"] = quarter_match.group(1).upper()
        
    return metadata


def load_and_chunk_pdfs():
    """
    Loads all PDFs from the company folders and applies RecursiveCharacterTextSplitter.
    """
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Initialize text splitter with configured chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    
    print("\n" + "=" * 50)
    print("STEP 1: Loading and Chunking Financial PDF Reports")
    print("=" * 50)
    
    for company in COMPANIES:
        company_path = os.path.join(DATA_DIR, company)
        if not os.path.exists(company_path):
            print(f"⚠️ Directory missing: {company_path}")
            continue
            
        pdf_files = [f for f in os.listdir(company_path) if f.lower().endswith(".pdf")]
        print(f"\n📂 Company: {company} ({len(pdf_files)} PDFs)")
        
        for filename in sorted(pdf_files):
            file_path = os.path.join(company_path, filename)
            file_meta = extract_metadata_from_filename(filename, company)
            
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Enrich each document page with custom metadata
                for doc in documents:
                    doc.metadata.update(file_meta)
                
                # Split documents into chunks
                chunks = text_splitter.split_documents(documents)
                print(f"  ✓ {filename:<30} -> {len(documents):>3} pages -> {len(chunks):>4} chunks")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                
    return all_chunks


def store_in_qdrant(chunks, batch_size=100):
    """
    Embeds the chunks using BAAI/bge-small-en-v1.5 and stores them in Qdrant Cloud.
    """
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in your .env file!")
        
    print("\n" + "=" * 50)
    print("STEP 2: Initializing BGE Embeddings Model")
    print("=" * 50)
    print("Model: BAAI/bge-small-en-v1.5 (Dimension: 384)")
    
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("\n" + "=" * 50)
    print("STEP 3: Connecting to Qdrant Cloud & Uploading Vectors")
    print("=" * 50)
    print(f"Target Collection: '{QDRANT_COLLECTION}'")
    print(f"Cluster URL: {QDRANT_URL}")
    print(f"Vector Name: 'dense' (384 dims)")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Initialize QdrantVectorStore targeting the 'dense' named vector
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        vector_name="dense"
    )
    
    total_chunks = len(chunks)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    print(f"\nUploading {total_chunks} chunks in {total_batches} batches (batch size: {batch_size})...\n")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        qdrant_store.add_documents(batch)
        print(f"  ✓ Uploaded batch {batch_num}/{total_batches} ({min(i + batch_size, total_chunks)}/{total_chunks} chunks)")
        
    # Verify final count in collection
    collection_info = client.get_collection(QDRANT_COLLECTION)
    count_info = client.count(collection_name=QDRANT_COLLECTION)
    
    print("\n" + "=" * 50)
    print("🎉 INGESTION COMPLETE!")
    print("=" * 50)
    print(f"Total points now in '{QDRANT_COLLECTION}': {count_info.count}")
    print(f"Collection status: {collection_info.status}")
    return qdrant_store


if __name__ == "__main__":
    print("🚀 Starting Document Ingestion into Qdrant Cloud...")
    chunks = load_and_chunk_pdfs()
    
    print(f"\nTotal chunks created across all companies: {len(chunks)}")
    
    if chunks:
        print("\n📄 Sample Chunk Preview:")
        print("-" * 40)
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Content:\n{chunks[0].page_content[:300]}...")
        print("-" * 40)
        
        # Store in Vector Database
        store_in_qdrant(chunks, batch_size=100)
    else:
        print("\n⚠️ No PDFs found! Please check your Data folder.")
