# AI Tutor Agent - PHP 1510/2510

An AI-powered learning assistant for **Principles of Biostatistics and Data Analysis** (PHP 1510/2510) at Brown University.

## 🎯 Goal

An AI agent that:
- ✅ Engages students in conversation about statistical concepts
- ✅ Assesses conceptual understanding through guided Q&A (not grading)
- ✅ Helps students apply concepts to research articles
- ✅ Adapts to student skill level with hints and resources
- ✅ Supports course learning objectives

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   - Create `.env` file with `OPENAI_API_KEY=your_key`
   - (Optional) Add `CANVAS_API_TOKEN` for Canvas integration

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

   The app will use the existing index automatically.

4. **To update/add materials:**
   - Add files to: `docs/`, `assessments/`, `textbook/`, `data/`, `articles/`
   - See [MANUAL_SETUP.md](MANUAL_SETUP.md) for detailed instructions
   - Run: `python scripts/ingest_docs_safe.py` to rebuild index

📖 **See [SETUP.md](SETUP.md) for detailed setup instructions.**

## 📚 Supported Sources

| Source | Description | Location |
|--------|-------------|----------|
| **Textbook** | Mathematical Statistics with Resampling and R (3rd ed.) | `textbook/` |
| **Lecture Slides** | Professor Lipman's slides and handouts | `docs/` |
| **Assessments** | Homework, quizzes, exams, solutions | `assessments/` |
| **Data** | CSV/Excel datasets for labs | `data/` |
| **Articles** | Journal articles from BruKnow, Nature, etc. | `articles/` |

## 💬 Features

### Three Interaction Modes

1. **💬 Conversation Mode**
   - Ask questions about concepts
   - Get explanations based on course materials
   - Receive clarification prompts

2. **📝 Assessment Mode**
   - Practice with guided questions
   - Get feedback on answers
   - Focus on understanding, not just answers

3. **📄 Article Analysis Mode**
   - Analyze research articles
   - Apply course concepts to real research
   - Get help interpreting statistical methods

### Additional Features

- 🔍 **Source Filtering**: Filter by type (textbook, slides, assessments, etc.)
- 📊 **Adaptive Difficulty**: Beginner/Intermediate/Advanced levels
- 💾 **Chat History**: Save and continue conversations
- 📄 **Export to PDF**: Save chat sessions for review

## 🛠️ Project Structure

```
agentTutor/
├── app.py                      # Main Streamlit application
├── scripts/
│   └── ingest_docs_safe.py     # Document ingestion system
├── docs/                        # Lecture slides and handouts
├── assessments/                 # Homework, quizzes, solutions
├── data/                        # Data files (CSV, Excel)
├── textbook/                    # Textbook chapters
├── articles/                    # Journal articles
└── index/                       # Generated search index (index.pkl)
```

## 🔗 Links

- **Live App**: [https://agenttutor.streamlit.app](https://agenttutor.streamlit.app)
- **Canvas**: PHP 1510/2510 Course Page
- **BruKnow Library**: [Search for articles](https://bruknow.library.brown.edu/discovery/search?vid=01BU_INST:BROWN)
- **Nature.com**: [Public health articles](https://www.nature.com)

## 📝 Course Learning Objectives Supported

- ✅ Explain fundamental concepts of statistics and their applications in public health
- ✅ Demonstrate written communication skills to explain statistical findings clearly

## 🤝 Contributing

This project is for Professor Lipman's PHP 1510/2510 class. For questions or issues:
- Check the [SETUP.md](SETUP.md) guide
- Contact course staff via EdStem
- Submit issues on GitHub

## 📄 License

Educational use for PHP 1510/2510 students and faculty.
