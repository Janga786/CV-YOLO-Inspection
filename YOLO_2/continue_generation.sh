#!/bin/bash
# Simple script to continue generation in a loop

echo "🔄 Starting continuous generation..."
echo "Press Ctrl+C to stop"

while true; do
    echo "▶️  Running generation batch..."
    python3 resume_generation.py
    
    # Check if we reached the target
    image_count=$(ls synthetic_dataset/images/*.png 2>/dev/null | wc -l)
    echo "📊 Current count: $image_count images"
    
    if [ "$image_count" -ge 3000 ]; then
        echo "🎉 Target reached! Generated $image_count images"
        break
    fi
    
    echo "⏳ Waiting 5 seconds before next batch..."
    sleep 5
done

echo "✅ Generation complete!"