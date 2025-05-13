"""
Usage:
python del_checkpoint.py --model_dir "/path/to/checkpoints" --keep_last_percent 10 --middle_interval_percent 7 --early_interval_percent 15 --dry_run
"""

import os
import glob
import re
import argparse
from datetime import datetime

def get_model_checkpoints(model_dir):
    pattern = re.compile(r"model_step_(\d+)\.pt")
    model_files = []
    
    for filename in glob.glob(os.path.join(model_dir, "model_step_*.pt")):
        match = pattern.search(os.path.basename(filename))
        if match:
            step = int(match.group(1))
            model_files.append((filename, step))
    
    model_files.sort(key=lambda x: x[1])
    return model_files
def calculate_checkpoints_to_keep(model_files, keep_last_percent=5, 
                                 middle_interval_percent=5, early_interval_percent=10):
    """Calculate checkpoints to keep using percentages instead of hardcoded steps"""
    if not model_files:
        return set()
        
    to_keep = set()
    
    # Always keep first and last checkpoints
    to_keep.add(model_files[0][1])
    to_keep.add(model_files[-1][1])
    
    last_step = model_files[-1][1]
    first_step = model_files[0][1]
    total_range = last_step - first_step
    
    # Keep all checkpoints in last N%
    keep_last_threshold = last_step - (total_range * keep_last_percent / 100)
    for _, step in model_files:
        if step > keep_last_threshold:
            to_keep.add(step)
    
    # Calculate middle phase interval (as percentage of total range)
    middle_start = first_step + (total_range * 1/3)
    middle_end = keep_last_threshold
    middle_interval = max(1, int(total_range * middle_interval_percent / 100))
    
    for step in range(int(middle_start), int(middle_end), middle_interval):
        closest_step = find_closest_step(model_files, step)
        if closest_step is not None:
            to_keep.add(closest_step)
    
    # Calculate early phase interval (as percentage of total range)
    early_end = middle_start
    early_interval = max(1, int(total_range * early_interval_percent / 100))
    
    for step in range(first_step, int(early_end), early_interval):
        closest_step = find_closest_step(model_files, step)
        if closest_step is not None:
            to_keep.add(closest_step)
            
    return to_keep

def find_closest_step(model_files, target_step):
    """Find the actual checkpoint closest to target step"""
    steps = [step for _, step in model_files]
    closest = None
    min_diff = float('inf')
    
    for step in steps:
        diff = abs(step - target_step)
        if diff < min_diff:
            min_diff = diff
            closest = step
            
    return closest

def main():
    parser = argparse.ArgumentParser(description='Model checkpoint cleanup tool')
    parser.add_argument('--model_dir', type=str, default=None, 
                        help='Directory containing model checkpoints')
    parser.add_argument('--keep_last_percent', type=float, default=5, 
                        help='Keep all checkpoints in last N% of training')
    parser.add_argument('--middle_interval_percent', type=float, default=5, 
                        help='Middle phase keep interval (% of total range)')
    parser.add_argument('--early_interval_percent', type=float, default=10, 
                        help='Early phase keep interval (% of total range)')
    parser.add_argument('--dry_run', action='store_true', 
                        help='Show files to be deleted without actually deleting')
    parser.add_argument('--no_confirm', action='store_true', 
                        help='Delete without confirmation')
    
    args = parser.parse_args()
    
    # If no directory specified, use newest checkpoint directory
    if args.model_dir is None:
        checkpoint_dirs = [d for d in glob.glob("**/checkpoints/*", recursive=True) 
                          if os.path.isdir(d)]
        if not checkpoint_dirs:
            print("No checkpoint directory found!")
            return
        # Sort by modification time, pick newest
        newest_dir = max(checkpoint_dirs, key=os.path.getmtime)
        args.model_dir = newest_dir
        print(f"Auto-selected newest checkpoint directory: {args.model_dir}")
    
    model_files = get_model_checkpoints(args.model_dir)
    
    if not model_files:
        print(f"No checkpoint files found in {args.model_dir}")
        return
        
    # Find latest checkpoint
    latest_checkpoint = model_files[-1]
    print(f"Found latest checkpoint: {os.path.basename(latest_checkpoint[0])} (step: {latest_checkpoint[1]})")
    
    to_keep = calculate_checkpoints_to_keep(
        model_files, 
        args.keep_last_percent,
        args.middle_interval_percent,
        args.early_interval_percent
    )
    
    files_to_delete = [(filename, step) for filename, step in model_files if step not in to_keep]
    
    # Print info and confirmation
    print(f"Found {len(model_files)} model checkpoint files.")
    print(f"Planning to keep {len(to_keep)} files, delete {len(files_to_delete)} files.")
    
    print("\nFiles to keep:")
    for filename, step in model_files:
        if step in to_keep:
            print(f"  {os.path.basename(filename)}")
    
    print("\nFiles to delete:")
    for filename, step in files_to_delete:
        print(f"  {os.path.basename(filename)}")
    
    # Create backup log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.model_dir, f"checkpoint_cleanup_{timestamp}.log")
    
    with open(log_file, 'w') as f:
        f.write(f"Checkpoint cleanup log - {timestamp}\n")
        f.write(f"Directory: {args.model_dir}\n")
        f.write(f"Total files: {len(model_files)}\n")
        f.write(f"Files kept: {len(to_keep)}\n")
        f.write(f"Files deleted: {len(files_to_delete)}\n\n")
        
        f.write("Kept files:\n")
        for filename, step in model_files:
            if step in to_keep:
                f.write(f"  {os.path.basename(filename)}\n")
                
        f.write("\nDeleted files:\n")
        for filename, step in files_to_delete:
            f.write(f"  {os.path.basename(filename)}\n")
    
    print(f"\nCleanup plan saved to: {log_file}")
    
    if args.dry_run:
        print("\nDry run complete - no files were deleted.")
        return
        
    # Safety check - confirmation
    if not args.no_confirm:
        confirmation = input("\nContinue with deletion? (yes/no): ")
        if confirmation.lower() != "yes":
            print("Deletion cancelled.")
            return
    
    deleted_count = 0
    error_count = 0
    
    for filename, _ in files_to_delete:
        try:
            os.remove(filename)
            print(f"Deleted: {os.path.basename(filename)}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {os.path.basename(filename)}: {e}")
            error_count += 1
    
    print(f"\nDeletion complete. Deleted {deleted_count} files, {error_count} errors.")
    
    # Update log
    with open(log_file, 'a') as f:
        f.write(f"\nActual deletions: {deleted_count} files\n")
        f.write(f"Deletion errors: {error_count} files\n")
        f.write(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
