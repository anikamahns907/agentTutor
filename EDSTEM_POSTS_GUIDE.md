# EdStem Posts Gathering Guide

## Quick Method: Individual Posts

### Option 1: Use the Post Processor Script (Recommended)

The script helps you organize individual posts:

```bash
python scripts/edstem_post_processor.py
```

This will:
1. Prompt you to paste each post
2. Ask for metadata (title, author, topic, date)
3. Save posts as formatted text files in `docs/edstem_posts/`
4. Automatically organize them for ingestion

**How to use:**
1. Go to EdStem
2. Open a discussion post
3. Copy the entire post (question + answer)
4. Run the script
5. Paste the post when prompted
6. Fill in metadata (title, author, topic)
7. Repeat for each important post

### Option 2: Manual Copy/Paste

1. **Create folder:**
   ```bash
   mkdir -p docs/edstem_posts
   ```

2. **For each important post:**
   - Copy the post text from EdStem
   - Create a text file: `docs/edstem_posts/topic_post_name.txt`
   - Paste the content
   - Save

3. **Format example:**
   ```
   Title: Understanding Confidence Intervals
   Posted by: Professor Lipman
   Date: 2025-10-15
   Topic: Confidence Intervals
   
   [Post content here]
   
   [Replies/answers if relevant]
   ```

## What Posts to Gather

### High-Value Posts:

1. **Instructor/TA Explanations**
   - Clear explanations of concepts
   - Step-by-step solutions
   - Common misconceptions clarified

2. **Conceptual Questions**
   - Questions that get detailed answers
   - Questions about reasoning (not just calculations)
   - "Why" questions

3. **Problem-Solving Discussions**
   - How to approach a problem
   - Different solution methods
   - Common mistakes

4. **Real-World Applications**
   - Connections to public health
   - Article discussions
   - Case studies

### Skip These:

- Simple "yes/no" questions
- Administrative questions
- Questions answered with just a link
- Very short responses

## Organizing Posts by Topic

### Suggested File Naming:

```
docs/edstem_posts/
├── confidence_intervals_discussion.txt
├── p_values_explanation.txt
├── hypothesis_testing_qa.txt
├── sampling_distributions.txt
├── regression_interpretation.txt
└── ...
```

### Topics to Look For:

- Confidence Intervals
- Hypothesis Testing
- p-values
- Sampling Distributions
- Regression Analysis
- Statistical Significance
- Type I/II Errors
- Power Analysis
- Resampling Methods
- Public Health Applications

## Quick Workflow

1. **Browse EdStem discussions**
   - Look for posts with detailed explanations
   - Focus on instructor/TA responses

2. **Copy the post**
   - Select all text (question + answer)
   - Copy to clipboard

3. **Use the processor:**
   ```bash
   python scripts/edstem_post_processor.py
   ```
   - Paste post
   - Add title/topic
   - Save

4. **After gathering posts:**
   ```bash
   python scripts/ingest_docs_safe.py
   ```
   This will process all .txt files in `docs/edstem_posts/`

## Example: Gathering a Good Post

**In EdStem, you see:**
```
Student: "I'm confused about when to use a t-test vs z-test. 
Can someone explain?"

TA Response: "Great question! The key difference is..."
[Detailed explanation with examples]
```

**What to do:**
1. Copy the entire thread (question + answer)
2. Run: `python scripts/edstem_post_processor.py`
3. Enter:
   - Title: "t-test vs z-test"
   - Author: TA Name
   - Topic: "Hypothesis Testing"
4. Paste the post content
5. Save

## Tips

- **Focus on quality over quantity**: 10 good explanations > 50 random posts
- **Organize by topic**: Makes it easier to find relevant content
- **Include context**: Save the question AND the answer
- **Note important clarifications**: Especially common misconceptions
- **Batch process**: Gather 5-10 posts, then run ingestion

## After Gathering

Once you have posts saved:

1. **Check your files:**
   ```bash
   ls -lh docs/edstem_posts/
   ```

2. **Update the index:**
   ```bash
   python scripts/ingest_docs_safe.py
   ```

3. **Test the agent:**
   ```bash
   streamlit run app.py
   ```

4. **Ask questions** related to the EdStem topics you added

## Alternative: Bulk Export (If Available)

If EdStem has an export feature:
1. Export discussions as text/CSV
2. Parse the export file
3. Save important posts using the processor script

## File Format

Posts are saved as plain text files that the ingestion script can process. The format includes:

- Title and metadata
- Post content
- Replies (if relevant)
- Topic tags

This makes posts searchable and usable by the AI agent!

