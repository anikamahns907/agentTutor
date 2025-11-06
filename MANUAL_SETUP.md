# Manual File Organization Guide

Since Canvas and EdStem API access is restricted, here's how to manually organize your course materials.

## 📁 Folder Structure

Create these folders in your project root:

```
agentTutor/
├── docs/              # Lecture slides and handouts
├── assessments/       # Homework, quizzes, exams, solutions
├── data/              # CSV, Excel files for labs
├── textbook/          # Textbook chapters (PDFs)
└── articles/         # Journal articles (PDFs)
```

## 📚 Step-by-Step: Organizing Canvas Files

### 1. Download from Canvas

1. **Go to your Canvas course** (PHP 1510/2510)
2. **Navigate through modules** or **Files** section
3. **Download files** to your computer

### 2. Organize by Type

As you download files, **organize them immediately** into the correct folders:

#### 📄 **Lecture Slides & Handouts** → `docs/`
- Files with names like:
  - "Week 1 Lecture Slides"
  - "Handout - Sampling"
  - "Lecture Notes"
  - "PHP 2510 Week 4 Handout.pdf" (already have this one!)
  
**Action:** Place all lecture-related PDFs in `docs/`

#### 📝 **Assessments** → `assessments/`
- Files with names like:
  - "Homework 1"
  - "Quiz 2 Solutions"
  - "Exam Study Guide"
  - "Practice Problems"
  - "Assignment 3"
  
**Action:** Place all homework, quiz, exam, and solution PDFs in `assessments/`

#### 📊 **Data Files** → `data/`
- Files with extensions:
  - `.csv`
  - `.xlsx`
  - `.xls`
  - Files with names like "lab_data", "dataset", etc.
  
**Action:** Place all data files in `data/`

#### 📖 **Textbook Chapters** → `textbook/`
- Files related to:
  - "Mathematical Statistics with Resampling and R"
  - Chapter PDFs
  - Textbook excerpts
  
**Action:** Place textbook-related PDFs in `textbook/`

### 3. Quick Organization Tips

**Option A: Download to a temp folder first**
```bash
# Create a temp download folder
mkdir ~/Downloads/canvas_temp

# Download all files there
# Then organize into project folders
```

**Option B: Download directly to correct folder**
- Open Canvas in one window
- Your project folder in another
- Drag files directly to the right folder as you download

## 📋 Checklist for Canvas Materials

Go through your Canvas course and download:

- [ ] **All lecture slides** (current semester)
- [ ] **All handouts** (any PDFs from lectures)
- [ ] **All homework assignments** (PDFs)
- [ ] **All quiz/exam questions** (if available)
- [ ] **All solution sets** (if available)
- [ ] **All datasets** (CSV/Excel files used in labs)
- [ ] **Syllabus** (goes in `docs/`)
- [ ] **Learning objectives** (goes in `docs/`)

## 📰 Step-by-Step: Organizing EdStem Files

EdStem typically has:
- Discussion posts (text - can copy/paste if needed)
- Shared resources
- Assignment files

### For EdStem:

1. **Download any shared files** → organize like Canvas files above
2. **Copy important discussion content** → save as text files in `docs/` if relevant
3. **Download any assignment files** → place in `assessments/`

## 📄 Step-by-Step: Adding Journal Articles

### From BruKnow Library

1. **Search**: https://bruknow.library.brown.edu
2. **Keywords to try:**
   - "biostatistics public health"
   - "vaccines statistics"
   - "nature biostatistics"
   - "public health epidemiology"
3. **Download PDFs** of relevant articles
4. **Save to**: `articles/` folder
5. **Name files clearly**: e.g., `nature_vaccine_study_2023.pdf`

### From Nature.com

1. **Access via Brown login** (automatic)
2. **Search** for public health articles
3. **Download PDFs**
4. **Save to**: `articles/` folder

### From Other Sources

- Any research article PDFs → `articles/`
- ArXiv papers → `articles/`
- Other journal articles → `articles/`

## 🔄 After Organizing Files

Once you've organized all files:

1. **Build the index:**
   ```bash
   python scripts/ingest_docs.py
   ```

2. **Verify what was loaded:**
   - The script will show you how many files from each folder
   - Check the summary at the end

3. **Test the agent:**
   ```bash
   streamlit run app.py
   ```

## 📝 Example File Organization

Here's what your folders might look like:

```
docs/
  ├── Week_1_Lecture_Slides.pdf
  ├── Week_2_Lecture_Slides.pdf
  ├── PHP 2510 Week 4 Handout.pdf
  ├── Sampling_Handout.pdf
  └── Syllabus.pdf

assessments/
  ├── Homework_1.pdf
  ├── Homework_1_Solutions.pdf
  ├── Quiz_2.pdf
  └── Midterm_Study_Guide.pdf

data/
  ├── lab_dataset_1.csv
  ├── sample_data.xlsx
  └── exercise_data.csv

textbook/
  ├── Chapter_1_Introduction.pdf
  ├── Chapter_5_Resampling.pdf
  └── Chapter_8_Hypothesis_Testing.pdf

articles/
  ├── nature_vaccine_effectiveness.pdf
  ├── bruknow_biostat_review.pdf
  └── arxiv_statistical_methods.pdf
```

## 🎯 Quick Reference

| If you see... | Put it in... |
|--------------|--------------|
| "Lecture", "Slides", "Handout" | `docs/` |
| "Homework", "Quiz", "Exam", "Solution" | `assessments/` |
| `.csv`, `.xlsx`, "data", "dataset" | `data/` |
| "Chapter", "Textbook", "Book" | `textbook/` |
| Research paper, journal article | `articles/` |
| Not sure? | `docs/` (default) |

## ✅ Verification

After organizing, run:

```bash
python scripts/ingest_docs.py
```

You should see output like:
```
📚 Starting document ingestion...

📄 Loading Lecture slides and handouts from docs/
  ✓ Loaded Week_1_Lecture_Slides.pdf (25 pages)
  ✓ Loaded PHP 2510 Week 4 Handout.pdf (5 pages)

📝 Loading Assessment solutions from assessments/
  ✓ Loaded Homework_1.pdf (3 pages)
  ...

✅ Created 250 text chunks
✅ Index saved to index/index.pkl
```

## 🔄 Regular Updates

**Weekly routine:**
1. Download new lecture slides → `docs/`
2. Download new homework → `assessments/`
3. Download new datasets → `data/`
4. Run `python scripts/ingest_docs.py` to update index
5. (Optional) Restart Streamlit app if needed

## 💡 Tips

- **Name files clearly** so you can identify them later
- **Keep original Canvas structure** in mind when organizing
- **Don't worry about duplicates** - the ingestion script handles them
- **Start with essentials** - you can always add more later
- **Test as you go** - add a few files, run ingestion, test the agent

## 🆘 Troubleshooting

**"No PDFs found"** → Check that files are actually PDFs and in the right folders

**"Error loading file"** → File might be corrupted, try re-downloading

**Missing content** → Some PDFs might be scanned images - they'll still work but may need OCR (not currently implemented)

**Files not showing up** → Make sure you ran `python scripts/ingest_docs.py` after adding files

