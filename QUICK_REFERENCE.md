# Quick Reference Card

## 📁 Where to Put Files

| File Type | Folder | Examples |
|-----------|--------|----------|
| Lecture slides | `docs/` | Week 1 Slides, Handouts |
| Homework/Quizzes | `assessments/` | HW1, Quiz 2, Solutions |
| Data files | `data/` | .csv, .xlsx files |
| Textbook | `textbook/` | Chapter PDFs |
| Articles | `articles/` | Research papers |

## 🔄 Workflow

```
1. Download files from Canvas/EdStem
   ↓
2. Organize into folders (docs/, assessments/, etc.)
   ↓
3. Run: python scripts/ingest_docs.py
   ↓
4. Run: streamlit run app.py
   ↓
5. Test the agent!
```

## 📝 Canvas Checklist

- [ ] All lecture slides → `docs/`
- [ ] All handouts → `docs/`
- [ ] All homework → `assessments/`
- [ ] All quizzes/exams → `assessments/`
- [ ] All solutions → `assessments/`
- [ ] All datasets → `data/`
- [ ] Syllabus → `docs/`

## 📚 After Adding Files

Always run:
```bash
python scripts/ingest_docs.py
```

This updates the searchable index with new files.

## 🎯 Quick Commands

```bash
# Build/update index
python scripts/ingest_docs.py

# Run the app
streamlit run app.py

# Install dependencies (if needed)
pip install -r requirements.txt
```

## 📖 Full Guides

- **Manual Setup**: [MANUAL_SETUP.md](MANUAL_SETUP.md)
- **Complete Setup**: [SETUP.md](SETUP.md)
- **Integration**: [INTEGRATION.md](INTEGRATION.md)

