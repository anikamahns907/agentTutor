# EdStem Data Gathering Guide

**Quick Start:** See [EDSTEM_EFFICIENT_GUIDE.md](EDSTEM_EFFICIENT_GUIDE.md) for the fastest method!

This guide helps you gather and organize materials from EdStem for the AI Tutor Agent.

## 📚 What to Look For in EdStem

EdStem typically contains:

### 1. **Discussion Posts** 📝
- Student questions and answers
- Conceptual clarifications
- Common misconceptions
- Problem-solving strategies

**How to gather:**
- Copy important discussion threads
- Save as text files or markdown
- Focus on posts that explain concepts clearly

### 2. **Course Notes/Resources** 📄
- Additional explanations
- Supplementary materials
- Code examples
- Study guides

**How to gather:**
- Download any PDFs or documents
- Copy text content into markdown files
- Save code snippets

### 3. **Assignment Files** 📋
- Problem sets
- Practice problems
- Solution discussions
- Grading rubrics

**How to gather:**
- Download PDFs or documents
- Copy problem descriptions
- Save solution explanations

### 4. **Announcements** 📢
- Course updates
- Important clarifications
- Study tips
- Resource links

**How to gather:**
- Copy important announcements
- Save links to external resources
- Note important dates/concepts

## 🔍 Step-by-Step: Gathering EdStem Data

### Step 1: Navigate EdStem

1. **Go to your EdStem course page**
2. **Look for sections:**
   - Discussions/Forum
   - Resources/Materials
   - Assignments
   - Announcements

### Step 2: Download Files

**For PDFs/Documents:**
- Click on file → Download
- Save to a temporary folder (e.g., `~/Downloads/edstem_temp`)
- Organize as you download

**For Text Content:**
- Select and copy text
- Paste into a text editor
- Save as `.txt` or `.md` file

### Step 3: Organize by Type

Create these folders in `edStemData/`:

```
edStemData/
├── discussions/     # Discussion threads, Q&A
├── resources/       # Additional resources, notes
├── assignments/     # EdStem-specific assignments
└── announcements/   # Important announcements
```

### Step 4: Process and Organize

**For each item, decide:**

| EdStem Content | Where it goes in agentTutor |
|----------------|----------------------------|
| Discussion explaining a concept | `docs/` (as a text file) |
| Assignment file (PDF) | `assessments/` |
| Practice problem | `assessments/` |
| Resource/study guide | `docs/` |
| Code examples | `docs/` (as text/markdown) |
| Dataset shared | `data/` |

## 📋 EdStem Checklist

Go through EdStem and gather:

- [ ] **Important Discussion Threads**
  - Questions about concepts
  - Explanations from instructor/TA
  - Common mistakes/clarifications
  
- [ ] **Resource Files**
  - Any PDFs or documents shared
  - Code examples
  - Study materials
  
- [ ] **Assignment Files**
  - Problem sets posted in EdStem
  - Practice problems
  - Solution discussions
  
- [ ] **Announcements**
  - Concept clarifications
  - Study tips
  - Important updates
  
- [ ] **Links to External Resources**
  - Note URLs for articles
  - Download linked PDFs
  - Save to `articles/` if research papers

## 💡 Tips for Gathering EdStem Content

### For Discussions:

1. **Look for high-value threads:**
   - Questions with detailed answers
   - Instructor explanations
   - Conceptual clarifications
   - Common misconceptions

2. **Save format:**
   ```
   File: edstem_discussion_week1_concepts.txt
   
   Topic: Understanding Confidence Intervals
   Posted by: [Name]
   Date: [Date]
   
   Question: [Student question]
   Answer: [Instructor/TA answer]
   ```

3. **Focus on:**
   - Clear explanations
   - Step-by-step solutions
   - Conceptual understanding
   - Real-world applications

### For Resources:

1. **Download everything:**
   - PDFs, documents, code files
   - Save with descriptive names

2. **Organize immediately:**
   - Don't let files accumulate
   - Move to correct folders as you download

3. **Convert formats if needed:**
   - HTML → Save as text
   - Images → Keep as-is (will be processed)
   - Code → Save as `.txt` or `.md`

### For Assignments:

1. **Get both:**
   - Problem statements
   - Solution discussions/approaches

2. **If no PDF:**
   - Copy problem text
   - Save as markdown file
   - Include any code snippets

## 🔄 After Gathering EdStem Data

1. **Organize files:**
   - Move files from `edStemData/` to appropriate folders:
     - PDFs → `docs/`, `assessments/`, etc.
     - Text files → `docs/`
     - Data files → `data/`

2. **Update index:**
   ```bash
   python scripts/ingest_docs.py
   ```

3. **Test the agent:**
   ```bash
   streamlit run app.py
   ```

## 📝 Example: Organizing EdStem Content

**Scenario:** You find a discussion thread about "Understanding p-values"

1. **Copy the thread content**
2. **Save as:** `docs/edstem_pvalues_discussion.txt`
3. **Include:**
   - The question
   - The answer/explanation
   - Any follow-up clarifications

**Scenario:** EdStem has a shared practice problem PDF

1. **Download the PDF**
2. **Move to:** `assessments/edstem_practice_problems.pdf`
3. **Run ingestion:** `python scripts/ingest_docs.py`

**Scenario:** Instructor shares a research article link

1. **Follow the link**
2. **Download the PDF** (if available)
3. **Save to:** `articles/nature_publication_2024.pdf`
4. **Run ingestion:** `python scripts/ingest_docs.py`

## 🎯 Quick Reference

**EdStem → AgentTutor Organization:**

```
EdStem Discussion Threads     → docs/edstem_discussion_*.txt
EdStem Resource PDFs           → docs/ or assessments/
EdStem Assignment Files        → assessments/
EdStem Code Examples           → docs/edstem_code_*.txt
EdStem Data Files             → data/
EdStem Article Links           → Download → articles/
```

## ✅ Final Steps

After gathering EdStem data:

1. ✅ All files organized in correct folders
2. ✅ Run `python scripts/ingest_docs.py`
3. ✅ Check that EdStem content appears in the agent
4. ✅ Test asking questions about EdStem-discussed topics

## 🆘 Troubleshooting

**"Can't download from EdStem"**
- Some content might be view-only
- Copy text manually instead
- Take screenshots if needed (will be processed as images)

**"Too much content to organize"**
- Start with most important discussions
- Focus on instructor/TA explanations
- Add more content incrementally

**"Unsure where to put something"**
- When in doubt, put in `docs/`
- You can always reorganize later
- The ingestion script will find it

---

**Remember:** The goal is to capture the valuable explanations, clarifications, and resources that complement the Canvas materials. Focus on quality over quantity!

