# Quick EdStem Gathering - 5 Minutes

## ⚡ Fastest Method

### Step 1: Use EdStem Search (2 min)

Search for these topics one by one:
- `confidence intervals`
- `p-values` 
- `hypothesis testing`
- `sampling distributions`

Open the **top 2-3 results** for each topic (ones with instructor/TA answers).

### Step 2: Quick Copy/Paste (2 min)

1. Create/use the collection file:
   ```bash
   bash scripts/edstem_quick_gather.sh
   ```

2. Open `docs/edstem_posts/edstem_collection.txt` in a text editor

3. For each good post, paste:
   ```
   ========================================
   TOPIC: [topic name]
   ========================================
   
   [Paste entire post - question + answer]
   
   
   ```

### Step 3: Process (1 min)

```bash
python scripts/ingest_docs_safe.py
```

Done! ✅

## 🎯 What to Look For

**Gather these:**
- ✅ Instructor/TA explanations
- ✅ Posts with "helpful" or upvotes
- ✅ Detailed answers (not just "yes/no")
- ✅ Conceptual clarifications

**Skip these:**
- ❌ Administrative questions
- ❌ Very short answers
- ❌ Date/deadline questions

## 💡 Pro Tip

**You only need 10-20 high-quality posts!** Focus on:
- Posts from Professor Lipman
- Posts from TAs
- Most upvoted posts
- Pinned/featured posts

Quality > Quantity!

## Alternative: Bulk Method

If EdStem has export:
1. Export discussions
2. Send me the file
3. I'll create a parser script

---

**See [EDSTEM_EFFICIENT_GUIDE.md](EDSTEM_EFFICIENT_GUIDE.md) for more detailed strategies.**

