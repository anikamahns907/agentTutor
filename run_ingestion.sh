#!/bin/bash
# Simple script to run document ingestion with output

cd "$(dirname "$0")"
source venv/bin/activate

echo "🚀 Starting document ingestion..."
echo "📁 Working directory: $(pwd)"
echo ""

# Run with unbuffered output
python -u scripts/ingest_docs_safe.py

echo ""
echo "✅ Script completed. Check output above for results."

