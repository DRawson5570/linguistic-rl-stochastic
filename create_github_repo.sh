#!/bin/bash

# Create GitHub Repository
# Run this script to create and push the repo to GitHub

echo "🚀 Creating GitHub Repository: linguistic-rl-stochastic"
echo ""

# Check if already has remote
if git remote get-url origin 2>/dev/null; then
    echo "✅ Remote already configured"
else
    echo "📝 Please create the repo manually:"
    echo ""
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: linguistic-rl-stochastic"
    echo "3. Description: Teaching AI to thrive in uncertainty through natural language"
    echo "4. Make it PUBLIC"
    echo "5. DON'T initialize with README (we already have one)"
    echo "6. Click 'Create repository'"
    echo ""
    echo "Then come back here and press Enter..."
    read -p ""
    
    # Add remote
    echo "Adding remote..."
    git remote add origin https://github.com/DRawson5570/linguistic-rl-stochastic.git
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "✅ DONE!"
echo ""
echo "🎉 Your repo is live at:"
echo "   https://github.com/DRawson5570/linguistic-rl-stochastic"
echo ""
echo "Next steps:"
echo "1. Share on Reddit r/MachineLearning"
echo "2. Tweet about it (tag AI influencers)"
echo "3. Post on Hacker News"
echo "4. Watch the stars roll in! ⭐"
