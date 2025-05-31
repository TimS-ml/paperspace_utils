"""
Usage:

# use percent
python del_checkpoint.py --model_dir "/notebooks/x-transformers/checkpoints/<run_name>" --early_interval 5000 --middle_interval 10000 --last_interval 2000 --middle_start_percent 33 --last_start_percent 90 --dry_run

# or use specific steps
python del_checkpoint.py --model_dir "/notebooks/x-transformers/checkpoints/<run_name>" --early_interval 5000 --middle_interval 10000 --last_interval 2000 --middle_start_steps 50000 --last_start_steps 100000 --dry_run
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

def calculate_checkpoints_to_keep(model_files, early_interval=5000, middle_interval=10000, 
                                  last_interval=2000, middle_start_percent=None, 
                                  last_start_percent=None, middle_start_steps=None, 
                                  last_start_steps=None):
    """Calculate checkpoints to keep using fixed intervals for three phases
    
    Args:
        model_files: List of (filepath, step) tuples
        early_interval: Fixed interval for early phase (in steps)
        middle_interval: Fixed interval for middle phase (in steps)
        last_interval: Fixed interval for last phase (in steps)
        middle_start_percent: Start point for middle phase (% of total range)
        last_start_percent: Start point for last phase (% of total range)
        middle_start_steps: Start point for middle phase (in steps)
        last_start_steps: Start point for last phase (in steps)
    """
    if not model_files:
        return set()
        
    to_keep = set()
    
    # Always keep first and last checkpoints
    first_step = model_files[0][1]
    last_step = model_files[-1][1]
    to_keep.add(first_step)
    to_keep.add(last_step)
    
    total_range = last_step - first_step
    
    # Calculate phase boundaries based on percent or steps
    if middle_start_steps is not None:
        middle_start = middle_start_steps
    elif middle_start_percent is not None:
        middle_start = first_step + (total_range * middle_start_percent / 100)
    else:
        # Default to 33%
        middle_start = first_step + (total_range * 0.33)
    
    if last_start_steps is not None:
        last_start = last_start_steps
    elif last_start_percent is not None:
        last_start = first_step + (total_range * last_start_percent / 100)
    else:
        # Default to 90%
        last_start = first_step + (total_range * 0.9)
    
    # Early phase with fixed interval (from first to middle_start)
    for step in range(first_step, int(middle_start), early_interval):
        if step != first_step:  # Skip first step as it's already added
            closest_step = find_closest_step(model_files, step)
            if closest_step is not None:
                to_keep.add(closest_step)
    
    # Middle phase with fixed interval (from middle_start to last_start)
    for step in range(int(middle_start), int(last_start), middle_interval):
        closest_step = find_closest_step(model_files, step)
        if closest_step is not None:
            to_keep.add(closest_step)
    
    # Last phase with fixed interval (from last_start to last step)
    for step in range(int(last_start), last_step, last_interval):
        if step != last_step:  # Skip last step as it's already added
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
    
    # Phase intervals
    parser.add_argument('--early_interval', type=int, default=5000, 
                        help='Early phase keep interval (fixed number of steps)')
    parser.add_argument('--middle_interval', type=int, default=10000, 
                        help='Middle phase keep interval (fixed number of steps)')
    parser.add_argument('--last_interval', type=int, default=2000, 
                        help='Last phase keep interval (fixed number of steps)')
    
    # Phase boundaries - percent
    parser.add_argument('--middle_start_percent', type=float, default=None,
                        help='Start point for middle phase (% of total range)')
    parser.add_argument('--last_start_percent', type=float, default=None,
                        help='Start point for last phase (% of total range)')
    
    # Phase boundaries - steps
    parser.add_argument('--middle_start_steps', type=int, default=None,
                        help='Start point for middle phase (specific step number)')
    parser.add_argument('--last_start_steps', type=int, default=None,
                        help='Start point for last phase (specific step number)')
    
    parser.add_argument('--dry_run', action='store_true', 
                        help='Show files to be deleted without actually deleting')
    parser.add_argument('--no_confirm', action='store_true', 
                        help='Delete without confirmation')
    
    args = parser.parse_args()
    
    # Default values if neither percent nor steps provided
    if args.middle_start_percent is None and args.middle_start_steps is None:
        args.middle_start_percent = 33
        
    if args.last_start_percent is None and args.last_start_steps is None:
        args.last_start_percent = 90
    
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
        early_interval=args.early_interval,
        middle_interval=args.middle_interval,
        last_interval=args.last_interval,
        middle_start_percent=args.middle_start_percent,
        last_start_percent=args.last_start_percent,
        middle_start_steps=args.middle_start_steps,
        last_start_steps=args.last_start_steps
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
        if confirmation.lower() not in ("yes", "y"):
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
