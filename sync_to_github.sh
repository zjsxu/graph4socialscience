#!/bin/bash

# Git synchronization script for graph4socialscience project
# Usage: ./sync_to_github.sh "commit message"

set -e

echo "🔄 Synchronizing project to GitHub..."

# Check if commit message provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide a commit message"
    echo "Usage: ./sync_to_github.sh \"your commit message\""
    exit 1
fi

COMMIT_MSG="$1"

# Add all changes
echo "📁 Adding all changes..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to commit"
    exit 0
fi

# Show status
echo "📊 Git status:"
git status --short

# Commit changes
echo "💾 Committing changes..."
git commit -m "$COMMIT_MSG"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Successfully synchronized to GitHub!"
echo "🌐 Repository: https://github.com/zjsxu/graph4socialscience"