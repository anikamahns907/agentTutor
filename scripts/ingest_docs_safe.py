"""
Safe document ingestion with better error handling and memory management.
This version processes files one at a time to avoid memory issues.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from dotenv import load_dotenv

# Import with error handling
try:
    import torch
except ImportError as e:
    print(f"❌ Error importing torch: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"❌ Error importing transformers: {e}")
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError as e:
    print(f"❌ Error importing langchain: {e}")
    sys.exit(1)

load_dotenv()

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global model variables
_tokenizer = None
_model = None

def load_model_safely():
    """Load model with error handling."""
    global _tokenizer, _model
    
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    
    print("\n" + "="*60)
    print("🤖 Loading AI Embedding Model")
    print("="*60)
    print("📥 Downloading model (first time only, ~90MB)...")
    print("   This may take 1-5 minutes depending on your connection.\n")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
        print("✅ Model loaded successfully!\n")
        return _tokenizer, _model
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def embed_batch(texts, tokenizer, model, batch_size=8):
    """Generate embeddings for a batch of texts."""
    if not texts:
        return np.array([])
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            # Tokenize
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                attention_mask = inputs['attention_mask']
                embeddings = outputs.last_hidden_state
                
                # Mean pooling with attention mask
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embs = (sum_embeddings / sum_mask).numpy()
            
            # Normalize
            norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            batch_embs = batch_embs / norms
            
            all_embeddings.append(batch_embs)
            
            # Clean up
            del inputs, outputs, embeddings, batch_embs
            gc.collect()
            
        except Exception as e:
            print(f"  ⚠️  Error processing batch {i//batch_size + 1}: {e}")
            # Create zero embeddings for failed batch
            batch_embs = np.zeros((len(batch), 384))
            all_embeddings.append(batch_embs)
    
    if all_embeddings:
        return np.vstack(all_embeddings)
    else:
        return np.array([])

def load_pdf_safely(pdf_path):
    """Load a single PDF file safely."""
    try:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"  ❌ Error loading {pdf_path.name}: {e}")
        return []

def main():
    """Main ingestion function."""
    base_dir = Path(__file__).parent.parent
    
    print("\n" + "="*60)
    print("📚 AI Tutor Agent - Document Ingestion")
    print("="*60)
    print("\n🚀 Starting document processing...\n")
    
    # Load model first
    tokenizer, model = load_model_safely()
    
    # Define source folders
    source_folders = {
        'docs': 'Lecture slides, handouts, and EdStem posts',
        'assessments': 'Assessment solutions',
        'data': 'Data files',
        'textbook': 'Textbook chapters',
        'articles': 'Journal articles'
    }
    
    all_docs = []
    
    # Load documents
    for folder_name, description in source_folders.items():
        folder_path = base_dir / folder_name
        
        if not folder_path.exists():
            continue
        
        if folder_name == 'data':
            print(f"\n{'─'*60}")
            print(f"📊 Processing {description}")
            print(f"{'─'*60}")
            # Handle data files
            for csv_file in folder_path.glob("*.csv"):
                try:
                    print(f"   📊 {csv_file.name}", end=" ... ", flush=True)
                    df = pd.read_csv(csv_file, nrows=1000)
                    summary = f"Dataset: {csv_file.name}\n"
                    summary += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                    summary += "First few rows:\n"
                    summary += df.head(10).to_string()
                    
                    doc = Document(
                        page_content=summary,
                        metadata={
                            'source': str(csv_file.name),
                            'source_type': 'data'
                        }
                    )
                    all_docs.append(doc)
                    print(f"✅ {df.shape[0]} rows")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            for excel_file in folder_path.glob("*.xlsx"):
                try:
                    print(f"   📊 {excel_file.name}", end=" ... ", flush=True)
                    df = pd.read_excel(excel_file, nrows=1000)
                    summary = f"Dataset: {excel_file.name}\n"
                    summary += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
                    summary += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                    summary += "First few rows:\n"
                    summary += df.head(10).to_string()
                    
                    doc = Document(
                        page_content=summary,
                        metadata={
                            'source': str(excel_file.name),
                            'source_type': 'data'
                        }
                    )
                    all_docs.append(doc)
                    print(f"✅ {df.shape[0]} rows")
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        else:
            # Handle PDFs and text files (for EdStem posts)
            print(f"\n{'─'*60}")
            print(f"📄 Processing {description}")
            print(f"{'─'*60}")
            
            # Find all PDFs and text files
            pdf_files = []
            txt_files = []
            
            for pdf_file in folder_path.rglob("*.pdf"):
                # Skip image-only folders
                parent = pdf_file.parent
                if parent != folder_path:  # In subdirectory
                    jpg_files = list(parent.glob("*.jpg")) + list(parent.glob("*.png"))
                    if len(jpg_files) > len(list(parent.glob("*.pdf"))) * 2:
                        continue  # Skip image-heavy folders
                pdf_files.append(pdf_file)
            
            # Find text files (EdStem posts)
            for txt_file in folder_path.rglob("*.txt"):
                txt_files.append(txt_file)
            
            total_files = len(pdf_files) + len(txt_files)
            if total_files:
                if pdf_files:
                    print(f"   Found {len(pdf_files)} PDF file(s)")
                if txt_files:
                    print(f"   Found {len(txt_files)} text file(s) (EdStem posts)")
                print()
            else:
                print(f"   ⏭️  No files found in {folder_name}/\n")
            
            # Process PDFs
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"   [{i}/{total_files}] 📄 {pdf_file.name}", end=" ... ", flush=True)
                docs = load_pdf_safely(pdf_file)
                if docs:
                    for doc in docs:
                        doc.metadata['source'] = str(pdf_file.name)
                        doc.metadata['source_type'] = folder_name
                    all_docs.extend(docs)
                    print(f"✅ {len(docs)} pages")
                else:
                    print("❌ Failed")
                gc.collect()
            
            # Process text files (EdStem posts)
            for i, txt_file in enumerate(txt_files, len(pdf_files) + 1):
                try:
                    print(f"   [{i}/{total_files}] 📝 {txt_file.name}", end=" ... ", flush=True)
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document from text file
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': str(txt_file.name),
                            'source_type': folder_name,
                            'file_type': 'edstem_post'
                        }
                    )
                    all_docs.append(doc)
                    print("✅")
                except Exception as e:
                    print(f"❌ {e}")
            
            gc.collect()
        
        gc.collect()
    
    if not all_docs:
        print("\n" + "="*60)
        print("⚠️  No documents found!")
        print("="*60)
        print("\nPlease add PDFs or data files to the appropriate folders:")
        for folder_name in source_folders.keys():
            print(f"   📁 {folder_name}/")
        print()
        return
    
    print("\n" + "="*60)
    print(f"✅ Successfully loaded {len(all_docs)} document(s)")
    print("="*60)
    
    # Split documents
    print("\n📝 Splitting documents into chunks...")
    print("   Breaking text into searchable pieces...", end=" ", flush=True)
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = splitter.split_documents(all_docs)
        texts = [c.page_content for c in chunks]
        metadata = [c.metadata for c in chunks]
        print(f"✅ {len(texts)} chunks created\n")
    except Exception as e:
        print(f"\n❌ Error splitting: {e}")
        return
    
    # Generate embeddings
    print("🔢 Generating AI embeddings...")
    print("   This may take a few minutes...", end=" ", flush=True)
    try:
        embs = embed_batch(texts, tokenizer, model, batch_size=8)
        print(f"✅ Complete! ({embs.shape[0]} embeddings, {embs.shape[1]} dimensions)\n")
    except Exception as e:
        print(f"\n❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save index
    print("💾 Saving index to disk...", end=" ", flush=True)
    index_dir = base_dir / "index"
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / "index.pkl"
    
    try:
        index_data = {
            "texts": texts,
            "embs": embs,
            "metadata": metadata
        }
        
        with open(index_path, "wb") as f:
            pickle.dump(index_data, f)
        
        file_size = os.path.getsize(index_path) / (1024 * 1024)  # MB
        print(f"✅ Saved! ({file_size:.1f} MB)\n")
        
        # Summary
        source_counts = {}
        for meta in metadata:
            source_type = meta.get('source_type', 'unknown')
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        print("="*60)
        print("📊 Summary by Source Type")
        print("="*60)
        for source_type, count in sorted(source_counts.items()):
            bar = "█" * min(50, int(count / max(source_counts.values()) * 50))
            print(f"   {source_type:15s} {bar:50s} {count:4d} chunks")
        
    except Exception as e:
        print(f"\n❌ Error saving index: {e}")
        return
    
    print("\n" + "="*60)
    print("🎉 Ingestion Complete!")
    print("="*60)
    print(f"\n✅ Index ready at: {index_path}")
    print(f"📊 Total: {len(texts)} searchable chunks from {len(all_docs)} documents")
    print("\n🚀 Next step: Run 'streamlit run app.py' to start the AI Tutor Agent!")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

