#!/usr/bin/env python3
"""
Script to merge results from two-phase processing.
This script combines the translated results from Phase 1 (local) and Phase 2 (remote) 
back into a single CSV file.
"""

import argparse
import os
import csv
import sys
from typing import Dict, List, Tuple

def read_csv_to_dict(file_path: str) -> Dict[str, List[str]]:
    """Read CSV file and return as dictionary with row ID as key"""
    rows_dict = {}
    
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
        try:
            header = next(reader)
            rows_dict['header'] = header
        except StopIteration:
            print(f"⚠ Empty CSV file: {file_path}")
            return rows_dict
        
        for row in reader:
            if row:  # Skip empty rows
                # Use first column as ID (assuming it's unique)
                row_id = row[0] if row else ""
                rows_dict[row_id] = row
    
    return rows_dict

def merge_csv_files(phase1_file: str, phase2_file: str, output_file: str, 
                   original_file: str = None) -> None:
    """
    Merge Phase 1 and Phase 2 results back into a single CSV file.
    
    Args:
        phase1_file: Path to Phase 1 output (local processing results)
        phase2_file: Path to Phase 2 output (remote processing results)
        output_file: Path to final merged output
        original_file: Path to original input file (for row ordering)
    """
    print(f"🔄 Merging Phase 1 and Phase 2 results...")
    print(f"   📁 Phase 1: {os.path.basename(phase1_file)}")
    print(f"   📁 Phase 2: {os.path.basename(phase2_file)}")
    print(f"   📁 Output:  {os.path.basename(output_file)}")
    
    # Read Phase 1 results (local processing)
    print(f"  📖 Reading Phase 1 results...")
    phase1_data = read_csv_to_dict(phase1_file)
    if not phase1_data:
        print(f"  ❌ Error: Could not read Phase 1 file: {phase1_file}")
        return
    
    # Read Phase 2 results (remote processing)
    print(f"  📖 Reading Phase 2 results...")
    phase2_data = read_csv_to_dict(phase2_file)
    if not phase2_data:
        print(f"  ❌ Error: Could not read Phase 2 file: {phase2_file}")
        return
    
    # Verify headers match
    if phase1_data['header'] != phase2_data['header']:
        print(f"  ⚠ Warning: Headers don't match between Phase 1 and Phase 2 files")
        print(f"    Phase 1: {phase1_data['header']}")
        print(f"    Phase 2: {phase2_data['header']}")
    
    # Use Phase 1 header
    header = phase1_data['header']
    
    # Merge the data
    merged_data = {}
    
    # Add Phase 1 data
    for row_id, row in phase1_data.items():
        if row_id != 'header':
            merged_data[row_id] = row
    
    # Add Phase 2 data (will overwrite any duplicates)
    for row_id, row in phase2_data.items():
        if row_id != 'header':
            merged_data[row_id] = row
    
    print(f"  📊 Merge statistics:")
    print(f"    - Phase 1 rows: {len(phase1_data) - 1}")  # Subtract header
    print(f"    - Phase 2 rows: {len(phase2_data) - 1}")  # Subtract header
    print(f"    - Total merged: {len(merged_data)}")
    
    # Determine row ordering
    if original_file and os.path.exists(original_file):
        print(f"  📖 Reading original file for row ordering...")
        original_data = read_csv_to_dict(original_file)
        if original_data:
            # Use original file order
            row_order = []
            for row_id, row in original_data.items():
                if row_id != 'header':
                    row_order.append(row_id)
        else:
            # Fallback to sorted order
            row_order = sorted(merged_data.keys())
    else:
        # Use sorted order
        row_order = sorted(merged_data.keys())
    
    # Write merged output
    print(f"  📝 Writing merged output...")
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE, 
                          escapechar='\\', lineterminator='\n')
        
        # Write header
        writer.writerow(header)
        
        # Write rows in order
        for row_id in row_order:
            if row_id in merged_data:
                writer.writerow(merged_data[row_id])
            else:
                print(f"  ⚠ Warning: Row ID {row_id} not found in merged data")
    
    print(f"  ✅ Successfully merged {len(merged_data)} rows to {os.path.basename(output_file)}")

def main():
    parser = argparse.ArgumentParser(description="Merge Phase 1 and Phase 2 translation results")
    parser.add_argument("--phase1", required=True, help="Phase 1 output file (local processing results)")
    parser.add_argument("--phase2", required=True, help="Phase 2 output file (remote processing results)")
    parser.add_argument("--output", required=True, help="Output file for merged results")
    parser.add_argument("--original", help="Original input file (for maintaining row order)")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.phase1):
        print(f"❌ Error: Phase 1 file not found: {args.phase1}")
        return 1
    
    if not os.path.exists(args.phase2):
        print(f"❌ Error: Phase 2 file not found: {args.phase2}")
        return 1
    
    if args.original and not os.path.exists(args.original):
        print(f"⚠ Warning: Original file not found: {args.original}")
        print(f"  Will use sorted row order instead")
        args.original = None
    
    try:
        merge_csv_files(args.phase1, args.phase2, args.output, args.original)
        print(f"\n🎉 Merge completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
