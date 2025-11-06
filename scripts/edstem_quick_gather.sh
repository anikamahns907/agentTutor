#!/bin/bash
# Quick helper script for gathering EdStem posts efficiently

echo "============================================================"
echo "📝 EdStem Quick Gather Helper"
echo "============================================================"
echo ""
echo "This script helps you quickly gather EdStem posts."
echo ""
echo "Recommended workflow:"
echo "  1. Use EdStem search to find high-value posts"
echo "  2. Copy posts into one file: docs/edstem_posts/edstem_collection.txt"
echo "  3. Run ingestion to process them"
echo ""

# Create the directory
mkdir -p docs/edstem_posts

# Check if file exists
if [ -f "docs/edstem_posts/edstem_collection.txt" ]; then
    echo "✅ Found existing collection file"
    echo "   File: docs/edstem_posts/edstem_collection.txt"
    echo ""
    wc -l docs/edstem_posts/edstem_collection.txt
    echo ""
    echo "💡 Tips:"
    echo "   - Add more posts to this file"
    echo "   - Separate posts with: ========================================"
    echo "   - Include topic names for better organization"
    echo ""
else
    echo "📄 Creating new collection file..."
    cat > docs/edstem_posts/edstem_collection.txt << 'EOF'
# EdStem Posts Collection
# Copy and paste EdStem posts below
# Separate each post with: ========================================

========================================
TOPIC: [Topic Name]
AUTHOR: [Name]
DATE: [Date]
========================================

[Paste post content here]


EOF
    echo "✅ Created: docs/edstem_posts/edstem_collection.txt"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Open the file in your text editor"
    echo "   2. Copy posts from EdStem and paste them"
    echo "   3. Separate posts with the ======== lines"
    echo "   4. Save the file"
    echo "   5. Run: python scripts/ingest_docs_safe.py"
    echo ""
fi

echo "🔍 Search Terms to Try in EdStem:"
echo "   - confidence intervals"
echo "   - p-values"
echo "   - hypothesis testing"
echo "   - sampling distributions"
echo "   - regression"
echo "   - statistical significance"
echo ""
echo "💡 Focus on:"
echo "   - Instructor/TA posts"
echo "   - Most upvoted/helpful posts"
echo "   - Detailed explanations"
echo "   - Conceptual questions"
echo ""

