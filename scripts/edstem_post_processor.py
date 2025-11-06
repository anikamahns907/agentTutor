"""
Helper script to process EdStem posts and organize them for the agent.
This helps you gather and format individual posts from EdStem discussions.
"""

import os
from pathlib import Path
from datetime import datetime

def create_post_document(post_text, metadata=None):
    """Create a formatted document from an EdStem post."""
    if metadata is None:
        metadata = {}
    
    # Format the post as a document
    doc_text = f"EdStem Discussion Post\n"
    doc_text += f"{'='*50}\n\n"
    
    if metadata.get('title'):
        doc_text += f"Title: {metadata['title']}\n"
    if metadata.get('author'):
        doc_text += f"Posted by: {metadata['author']}\n"
    if metadata.get('date'):
        doc_text += f"Date: {metadata['date']}\n"
    if metadata.get('topic'):
        doc_text += f"Topic: {metadata['topic']}\n"
    
    doc_text += f"\n{'-'*50}\n\n"
    doc_text += post_text
    
    if metadata.get('replies'):
        doc_text += f"\n\n{'-'*50}\nReplies:\n"
        for reply in metadata['replies']:
            doc_text += f"\n{reply.get('author', 'Unknown')}: {reply.get('text', '')}\n"
    
    return doc_text

def save_post_to_file(post_text, metadata, output_dir="docs/edstem_posts"):
    """Save a post to a text file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename from title or topic
    if metadata.get('title'):
        filename = metadata['title'].replace(' ', '_').replace('/', '_')[:50]
    elif metadata.get('topic'):
        filename = metadata['topic'].replace(' ', '_').replace('/', '_')[:50]
    else:
        filename = f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    filename = f"{filename}.txt"
    filepath = output_path / filename
    
    doc_text = create_post_document(post_text, metadata)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(doc_text)
    
    return filepath

def main():
    """Interactive mode to help gather EdStem posts."""
    print("="*60)
    print("📝 EdStem Post Processor")
    print("="*60)
    print("\nThis tool helps you save EdStem posts for the AI Tutor Agent.")
    print("\nFor each post, you'll provide:")
    print("  - The post text (copy/paste from EdStem)")
    print("  - Optional metadata (title, author, topic, etc.)")
    print("\n" + "="*60 + "\n")
    
    posts = []
    
    while True:
        print("\n📄 New Post")
        print("-"*60)
        
        # Get post text
        print("Paste the post text (press Enter twice when done):")
        post_lines = []
        empty_lines = 0
        while True:
            try:
                line = input()
                if line == "":
                    empty_lines += 1
                    if empty_lines >= 2:
                        break
                else:
                    empty_lines = 0
                    post_lines.append(line)
            except EOFError:
                break
        
        if not post_lines:
            break
        
        post_text = "\n".join(post_lines)
        
        # Get metadata
        metadata = {}
        print("\n📋 Metadata (optional - press Enter to skip):")
        
        title = input("Title/Topic: ").strip()
        if title:
            metadata['title'] = title
        
        author = input("Author: ").strip()
        if author:
            metadata['author'] = author
        
        topic = input("Topic/Category (e.g., 'Confidence Intervals', 'p-values'): ").strip()
        if topic:
            metadata['topic'] = topic
        
        date = input("Date (or press Enter for today): ").strip()
        if date:
            metadata['date'] = date
        else:
            metadata['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Save post
        filepath = save_post_to_file(post_text, metadata)
        print(f"\n✅ Saved to: {filepath}")
        
        # Ask if more
        more = input("\nAdd another post? (y/n): ").strip().lower()
        if more != 'y':
            break
    
    print("\n" + "="*60)
    print("✅ Done!")
    print("="*60)
    print(f"\n📁 Posts saved to: docs/edstem_posts/")
    print("\n📝 Next step: Run 'python scripts/ingest_docs_safe.py' to update the index")
    print("   (Note: This will process .txt files in docs/edstem_posts/)")

if __name__ == "__main__":
    main()

