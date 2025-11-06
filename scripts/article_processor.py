"""
Process journal articles from BruKnow, Nature.com, or other sources.
This script handles article URLs and PDFs for inclusion in the knowledge base.
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
import arxiv

load_dotenv()

def download_pdf_from_url(url, output_path):
    """Download a PDF from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return False

def process_nature_article(url):
    """Process a Nature.com article URL."""
    # Nature articles may require authentication or have different formats
    # This is a basic implementation
    print(f"Processing Nature article: {url}")
    # You might need to use selenium or playwright for articles behind paywalls
    return None

def process_bruknow_article(url):
    """Process a BruKnow library article."""
    print(f"Processing BruKnow article: {url}")
    # BruKnow links might redirect to publisher sites
    # You may need to handle authentication here
    return None

def process_arxiv_article(arxiv_id):
    """Process an arXiv article."""
    try:
        paper = arxiv.Search(id_list=[arxiv_id]).results().__next__()
        # Download PDF
        pdf_url = paper.pdf_url
        output_path = Path("articles") / f"{arxiv_id}.pdf"
        output_path.parent.mkdir(exist_ok=True)
        
        if download_pdf_from_url(pdf_url, output_path):
            print(f"✅ Downloaded {arxiv_id} to {output_path}")
            return str(output_path)
    except Exception as e:
        print(f"Error processing arXiv article: {e}")
    return None

def process_article_url(url):
    """Process an article URL and download if possible."""
    base_dir = Path(__file__).parent.parent
    articles_dir = base_dir / "articles"
    articles_dir.mkdir(exist_ok=True)
    
    if "arxiv.org" in url or "arxiv.org/abs" in url:
        # Extract arXiv ID
        arxiv_id = url.split("/")[-1].replace(".pdf", "")
        return process_arxiv_article(arxiv_id)
    elif "nature.com" in url:
        return process_nature_article(url)
    elif "bruknow.library.brown.edu" in url:
        return process_bruknow_article(url)
    else:
        # Try to download as PDF
        filename = url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        output_path = articles_dir / filename
        
        if download_pdf_from_url(url, output_path):
            print(f"✅ Downloaded article to {output_path}")
            return str(output_path)
    
    return None

def main():
    """Main function for processing articles."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python article_processor.py <article_url_or_arxiv_id>")
        print("\nExamples:")
        print("  python article_processor.py https://arxiv.org/abs/2301.00001")
        print("  python article_processor.py https://www.nature.com/articles/s41586-023-00000-0")
        return
    
    url_or_id = sys.argv[1]
    result = process_article_url(url_or_id)
    
    if result:
        print(f"\n✅ Article processed successfully!")
        print(f"   Saved to: {result}")
        print(f"\n📝 Next step: Run 'python scripts/ingest_docs.py' to update the index")
    else:
        print("\n❌ Could not process article. You may need to:")
        print("   1. Download the PDF manually")
        print("   2. Place it in the articles/ folder")
        print("   3. Run 'python scripts/ingest_docs.py'")

if __name__ == "__main__":
    main()

