"""
Helper script to organize downloaded Canvas files into the correct folders.
This script helps move files from canvasData/ to the proper agentTutor folders.
"""

import os
import shutil
from pathlib import Path

def organize_canvas_files():
    """Organize files from canvasData/ into proper folders."""
    base_dir = Path(__file__).parent.parent
    
    # Source folders
    canvas_data = base_dir / "canvasData"
    lecture_source = canvas_data / "lectureSlidesAndHandouts"
    assessment_source = canvas_data / "assessments"
    
    # Destination folders
    docs_dir = base_dir / "docs"
    assessments_dir = base_dir / "assessments"
    
    # Create destination folders if they don't exist
    docs_dir.mkdir(exist_ok=True)
    assessments_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    
    print("📁 Organizing Canvas files...\n")
    
    # Move lecture slides and handouts
    if lecture_source.exists():
        print(f"📄 Processing lecture slides and handouts from {lecture_source.name}/")
        
        # Move PDF files
        for pdf_file in lecture_source.glob("*.pdf"):
            dest = docs_dir / pdf_file.name
            if not dest.exists():
                shutil.copy2(pdf_file, dest)
                print(f"  ✓ Moved {pdf_file.name} → docs/")
                moved_count += 1
            else:
                print(f"  ⏭️  Skipped {pdf_file.name} (already exists)")
        
        # Move image folders (keep structure or flatten)
        for folder in lecture_source.iterdir():
            if folder.is_dir():
                # Option 1: Keep folder structure
                dest_folder = docs_dir / folder.name
                dest_folder.mkdir(exist_ok=True)
                
                for img_file in folder.glob("*.jpg"):
                    dest = dest_folder / img_file.name
                    if not dest.exists():
                        shutil.copy2(img_file, dest)
                        print(f"  ✓ Moved {folder.name}/{img_file.name} → docs/{folder.name}/")
                        moved_count += 1
    
    # Move assessments
    if assessment_source.exists():
        print(f"\n📝 Processing assessments from {assessment_source.name}/")
        
        for file in assessment_source.iterdir():
            if file.is_file():
                dest = assessments_dir / file.name
                if not dest.exists():
                    shutil.copy2(file, dest)
                    print(f"  ✓ Moved {file.name} → assessments/")
                    moved_count += 1
                else:
                    print(f"  ⏭️  Skipped {file.name} (already exists)")
    
    print(f"\n✅ Organization complete!")
    print(f"   - {moved_count} files moved")
    print(f"\n📝 Next step: Run 'python scripts/ingest_docs.py' to build the index")

if __name__ == "__main__":
    organize_canvas_files()

