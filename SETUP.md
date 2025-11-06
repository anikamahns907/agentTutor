# Setup Guide for AI Tutor Agent

This guide will help you set up the complete AI Tutor Agent system for PHP 1510/2510.

## Prerequisites

1. Python 3.10 or higher
2. Canvas API access (for downloading course materials)
3. OpenAI API key
4. Access to course Canvas and EdStem

## Step 1: Install Dependencies

```bash
# Create virtual environment (if not already created)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment Variables

Create a `.env` file in the project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for Canvas integration)
CANVAS_API_TOKEN=your_canvas_token_here
CANVAS_BASE_URL=https://canvas.brown.edu

# Optional (for EdStem integration - if needed)
EDSTEM_API_KEY=your_edstem_key_here
```

### Getting Canvas API Token (Optional - Only if API access is available)

If Canvas API access is available:

1. Log into Canvas (Brown University)
2. Go to **Account** → **Settings**
3. Scroll to **Approved Integrations**
4. Click **+ New Access Token**
5. Give it a purpose (e.g., "Tutor Agent")
6. Copy the token and add to `.env`

**Note:** If API access is restricted, skip this and use manual download (see Step 3).

## Step 3: Organize Course Materials

Create the following folder structure:

```
agentTutor/
├── docs/              # Lecture slides and handouts
├── assessments/       # Homework, quizzes, exams, solutions
├── data/             # CSV, Excel files for labs
├── textbook/         # Textbook chapters (PDFs)
└── articles/        # Journal articles (PDFs)
```

### Downloading from Canvas (Manual - Recommended)

Since Canvas API access may be restricted, **manually download and organize files**:

1. **Go to Canvas course** → Download files
2. **Organize by type:**
   - Lecture slides/handouts → `docs/`
   - Homework/quizzes/exams → `assessments/`
   - Data files (.csv, .xlsx) → `data/`
   - Textbook chapters → `textbook/`
3. **See [MANUAL_SETUP.md](MANUAL_SETUP.md) for detailed instructions**

### Alternative: Canvas API (If Available)

If you have Canvas API access, you can use the automated downloader:

```bash
# First, get your course ID from Canvas URL
# Example: https://canvas.brown.edu/courses/12345
# Course ID is: 12345

python scripts/canvas_downloader.py --course-id YOUR_COURSE_ID

# Preview what will be downloaded (dry run)
python scripts/canvas_downloader.py --course-id YOUR_COURSE_ID --dry-run
```

### Manual File Organization

If you prefer to download manually:

1. **Textbook**: Download PDFs of relevant chapters → place in `textbook/`
2. **Slides/Handouts**: Download from Canvas → place in `docs/`
3. **Assessments**: Download homework/quizzes → place in `assessments/`
4. **Data**: Download CSV/Excel files → place in `data/`
5. **Articles**: Download PDFs or use article processor → place in `articles/`

## Step 4: Process Articles

### From BruKnow Library

1. Search for articles on [BruKnow](https://bruknow.library.brown.edu)
2. Download PDFs manually
3. Place in `articles/` folder

### From Nature.com

1. Access articles (may require Brown login)
2. Download PDFs
3. Place in `articles/` folder

### From arXiv

```bash
python scripts/article_processor.py https://arxiv.org/abs/2301.00001
```

## Step 5: Build the Knowledge Index

Once all materials are in place, create the searchable index:

```bash
python scripts/ingest_docs.py
```

This will:
- Load all PDFs from `docs/`, `assessments/`, `textbook/`, `articles/`
- Process data files from `data/`
- Split documents into chunks
- Generate embeddings
- Save index to `index/index.pkl`

**Expected output:**
```
📚 Starting document ingestion...

📄 Loading Lecture slides and handouts from docs/
  ✓ Loaded PHP 2510 Week 4 Handout.pdf (5 pages)

📊 Loading Data files from data/
  ✓ Loaded dataset sample_data.csv

✅ Created 150 text chunks
✅ Index saved to index/index.pkl
```

## Step 6: Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Step 7: Using the Agent

### Conversation Mode 💬
- Ask questions about statistical concepts
- Get explanations based on course materials
- Receive clarification prompts

### Assessment Mode 📝
- Practice with questions similar to homework
- Get feedback on your answers
- Focus on conceptual understanding

### Article Analysis Mode 📄
- Upload or analyze research articles
- Apply course concepts to real research
- Get help interpreting statistical methods

### Features

- **Source Filtering**: Filter by textbook, slides, assessments, data, or articles
- **Student Level**: Adjust difficulty (beginner/intermediate/advanced)
- **Chat History**: Continue conversations
- **Export to PDF**: Save your chat sessions

## Troubleshooting

### "Index not found" error
- Run `python scripts/ingest_docs.py` first

### "No documents found" error
- Check that you have PDFs or data files in the folders
- Verify file paths are correct

### Canvas download fails
- Check your API token in `.env`
- Verify course ID is correct
- Some files may require manual download

### Embedding errors
- Make sure transformers library is installed
- Check that you have enough disk space for model downloads

## Next Steps

1. **Add more materials**: Regularly add new slides, assessments, and articles
2. **Re-run ingestion**: After adding new files, run `ingest_docs.py` again
3. **Customize prompts**: Edit `app.py` to adjust agent behavior
4. **Track usage**: Monitor how students use the agent (anonymized)

## Integration with Canvas/EdStem

### Canvas LTI Integration
To embed the agent in Canvas:
1. Set up Streamlit Cloud deployment
2. Create LTI app in Canvas
3. Configure as external tool

### EdStem Integration
- Share the Streamlit app URL
- Or embed as iframe in EdStem pages

## Support

For issues or questions, check:
- GitHub Issues: [Your repo URL]
- Course EdStem forum
- Contact Professor Lipman

