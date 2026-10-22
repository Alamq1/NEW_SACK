#!/bin/bash

# Quick Verification Script - Check if labels are in correct detection format

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        RICE BAG LABELS - FORMAT VERIFICATION SCRIPT                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📁 Project Root: $PROJECT_ROOT"
echo ""

# Function to check a label directory
check_labels() {
    local labels_dir="$1"
    local dataset_name="$2"
    
    if [ ! -d "$labels_dir" ]; then
        echo "  ⊘ Directory not found: $labels_dir"
        return
    fi
    
    label_files=$(find "$labels_dir" -name "*.txt" -type f)
    
    if [ -z "$label_files" ]; then
        echo "  ⊘ No label files found"
        return
    fi
    
    file_count=$(echo "$label_files" | wc -l)
    echo "  📋 Files: $file_count"
    
    # Sample first file
    first_file=$(echo "$label_files" | head -1)
    first_line=$(head -1 "$first_file" 2>/dev/null)
    
    if [ -z "$first_line" ]; then
        echo "  ⚠️  First label file is empty"
        return
    fi
    
    # Count fields in first line
    field_count=$(echo "$first_line" | wc -w)
    
    echo "  📝 Sample: $first_line"
    echo "  🔢 Fields per line: $field_count"
    
    # Validate format
    if [ "$field_count" -eq 5 ]; then
        echo "  ✅ Format: DETECTION (class cx cy w h)"
    elif [ "$field_count" -gt 5 ] && [ "$((field_count % 2))" -eq 1 ]; then
        echo "  ❌ Format: POLYGON (class x1 y1 x2 y2 ... xn yn) - WRONG!"
    else
        echo "  ⚠️  Format: UNKNOWN (${field_count} fields)"
    fi
    echo ""
}

# Check all datasets
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  RICE BAG 10 (Primary)                                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📍 Training Labels:"
check_labels "$PROJECT_ROOT/data/Rice Bag 10/train/labels" "Rice Bag 10 - Train"

echo "📍 Validation Labels:"
check_labels "$PROJECT_ROOT/data/Rice Bag 10/valid/labels" "Rice Bag 10 - Valid"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  RICE BAG DATASET (Backup)                                         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📍 Training Labels:"
check_labels "$PROJECT_ROOT/data/Rice Bag dataset/train/labels" "Rice Bag Dataset - Train"

echo "📍 Validation Labels:"
check_labels "$PROJECT_ROOT/data/Rice Bag dataset/valid/labels" "Rice Bag Dataset - Valid"

echo "📍 Test Labels:"
check_labels "$PROJECT_ROOT/data/Rice Bag dataset/test/labels" "Rice Bag Dataset - Test"

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY                                                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

all_labels="$PROJECT_ROOT/data/Rice Bag 10/train/labels/*.txt $PROJECT_ROOT/data/Rice Bag 10/valid/labels/*.txt $PROJECT_ROOT/data/Rice Bag dataset/train/labels/*.txt $PROJECT_ROOT/data/Rice Bag dataset/valid/labels/*.txt $PROJECT_ROOT/data/Rice Bag dataset/test/labels/*.txt"

total_files=$(find "$PROJECT_ROOT/data" -name "*.txt" -path "*/labels/*" -type f | wc -l)
echo "✓ Total label files: $total_files"

# Count unique field counts
field_counts=$(find "$PROJECT_ROOT/data" -name "*.txt" -path "*/labels/*" -type f -exec sh -c 'wc -w < "$1"' _ {} \; 2>/dev/null | sort | uniq -c | sort -rn)

echo "✓ Field count distribution:"
echo "$field_counts" | while read count fields; do
    if [ "$fields" -eq 5 ]; then
        echo "    $fields fields: $count files (✅ DETECTION FORMAT - CORRECT!)"
    else
        echo "    $fields fields: $count files (⚠️  May need conversion)"
    fi
done

echo ""
echo "🎯 Expected: All files should have exactly 5 fields (class cx cy w h)"
echo ""

# Check if conversion tool exists
if [ -f "$PROJECT_ROOT/convert_polygon_to_bbox.py" ]; then
    echo "🔧 Conversion tool available: convert_polygon_to_bbox.py"
else
    echo "⚠️  Conversion tool not found"
fi

echo ""
