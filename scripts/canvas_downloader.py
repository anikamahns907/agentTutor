"""
Canvas API integration for downloading course materials.
This script helps download files from Canvas courses.

Usage:
1. Get your Canvas API token from Canvas > Account > Settings > New Access Token
2. Set CANVAS_API_TOKEN and CANVAS_BASE_URL in .env
3. Run: python scripts/canvas_downloader.py --course-id YOUR_COURSE_ID
"""

import os
import requests
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CANVAS_API_TOKEN = os.getenv("CANVAS_API_TOKEN")
CANVAS_BASE_URL = os.getenv("CANVAS_BASE_URL", "https://canvas.brown.edu")  # Adjust if different

def download_file(url, filepath, headers):
    """Download a file from Canvas."""
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

def get_canvas_course_files(course_id, headers):
    """Get all files from a Canvas course."""
    files_url = f"{CANVAS_BASE_URL}/api/v1/courses/{course_id}/files"
    files = []
    
    while files_url:
        response = requests.get(files_url, headers=headers, params={'per_page': 100})
        if response.status_code == 200:
            data = response.json()
            files.extend(data)
            # Check for pagination
            links = response.links
            if 'next' in links:
                files_url = links['next']['url']
            else:
                files_url = None
        else:
            print(f"Error: {response.status_code}")
            break
    
    return files

def organize_file_by_type(filename):
    """Determine which folder a file should go to based on its name."""
    filename_lower = filename.lower()
    
    if any(keyword in filename_lower for keyword in ['homework', 'hw', 'quiz', 'exam', 'assessment', 'solution']):
        return 'assessments'
    elif any(keyword in filename_lower for keyword in ['slide', 'lecture', 'handout', 'notes']):
        return 'docs'
    elif any(keyword in filename_lower for keyword in ['data', '.csv', '.xlsx', 'dataset']):
        return 'data'
    elif any(keyword in filename_lower for keyword in ['textbook', 'chapter', 'book']):
        return 'textbook'
    elif any(keyword in filename_lower for keyword in ['article', 'paper', 'journal']):
        return 'articles'
    else:
        return 'docs'  # Default

def main():
    parser = argparse.ArgumentParser(description='Download files from Canvas course')
    parser.add_argument('--course-id', type=str, required=True, help='Canvas course ID')
    parser.add_argument('--output-dir', type=str, default='.', help='Base output directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without downloading')
    
    args = parser.parse_args()
    
    if not CANVAS_API_TOKEN:
        print("❌ Error: CANVAS_API_TOKEN not found in .env file")
        print("\nTo get your Canvas API token:")
        print("1. Go to Canvas > Account > Settings")
        print("2. Scroll down to 'Approved Integrations'")
        print("3. Click '+ New Access Token'")
        print("4. Add the token to your .env file as CANVAS_API_TOKEN=your_token")
        return
    
    headers = {
        'Authorization': f'Bearer {CANVAS_API_TOKEN}'
    }
    
    print(f"📚 Fetching files from Canvas course {args.course_id}...")
    files = get_canvas_course_files(args.course_id, headers)
    
    if not files:
        print("No files found in this course.")
        return
    
    print(f"Found {len(files)} files\n")
    
    base_dir = Path(args.output_dir).resolve()
    base_dir = base_dir.parent if base_dir.name == 'scripts' else base_dir
    
    organized = {
        'docs': [],
        'assessments': [],
        'data': [],
        'textbook': [],
        'articles': []
    }
    
    for file_info in files:
        filename = file_info.get('display_name', file_info.get('filename', 'unknown'))
        folder = organize_file_by_type(filename)
        organized[folder].append((file_info, filename))
    
    # Show what will be downloaded
    print("📁 Files organized by type:\n")
    for folder, file_list in organized.items():
        if file_list:
            print(f"  {folder}/ ({len(file_list)} files)")
            for file_info, filename in file_list[:5]:  # Show first 5
                print(f"    - {filename}")
            if len(file_list) > 5:
                print(f"    ... and {len(file_list) - 5} more")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - no files downloaded")
        return
    
    # Download files
    print("\n⬇️  Downloading files...\n")
    downloaded = 0
    skipped = 0
    
    for folder, file_list in organized.items():
        folder_path = base_dir / folder
        folder_path.mkdir(exist_ok=True)
        
        for file_info, filename in file_list:
            file_path = folder_path / filename
            
            if file_path.exists():
                print(f"  ⏭️  Skipped {filename} (already exists)")
                skipped += 1
                continue
            
            # Get download URL
            download_url = file_info.get('url')
            if not download_url:
                # Need to get download URL
                file_id = file_info['id']
                file_url = f"{CANVAS_BASE_URL}/api/v1/files/{file_id}"
                file_response = requests.get(file_url, headers=headers, params={'include[]': 'url'})
                if file_response.status_code == 200:
                    file_data = file_response.json()
                    download_url = file_data.get('url')
            
            if download_url:
                print(f"  ⬇️  Downloading {filename}...")
                if download_file(download_url, file_path, headers):
                    downloaded += 1
                    print(f"     ✅ Saved to {folder_path.name}/{filename}")
                else:
                    print(f"     ❌ Failed to download {filename}")
            else:
                print(f"  ❌ Could not get download URL for {filename}")
    
    print(f"\n✅ Download complete!")
    print(f"   - {downloaded} files downloaded")
    print(f"   - {skipped} files skipped (already exist)")
    print(f"\n📝 Next step: Run 'python scripts/ingest_docs.py' to update the index")

if __name__ == "__main__":
    main()

