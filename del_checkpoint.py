import os
import glob
import re

# Directory containing model files
model_dir = "/notebooks/x-transformers/checkpoints/250511_0433_lr_0.0001_bs_4"

# Pattern to match model files and extract step number
pattern = re.compile(r"model_step_(\d+)\.pt")

# Collect all model files and their step numbers
model_files = []
for filename in glob.glob(os.path.join(model_dir, "model_step_*.pt")):
    match = pattern.search(os.path.basename(filename))
    if match:
        step = int(match.group(1))
        model_files.append((filename, step))

# Sort by step number
model_files.sort(key=lambda x: x[1])

# Define which files to keep based on different retention strategies
# Keep:
# - Every file for the last 5000 steps (e.g., from 94000 to 99000)
# - Every 5000 steps from 50000 to 94000
# - Every 10000 steps from 1000 to 50000
# - Always keep the first and last checkpoint

to_keep = set()

# Always keep first and last checkpoint
to_keep.add(model_files[0][1])
to_keep.add(model_files[-1][1])

# Keep every file for the last 5000 steps
last_step = model_files[-1][1]
for filename, step in model_files:
    if step > last_step - 5000:
        to_keep.add(step)

# Keep every 5000 steps between 50000 and 94000
for step in range(50000, last_step - 5000, 5000):
    to_keep.add(step)

# Keep every 10000 steps between 1000 and 50000
for step in range(1000, 50000, 10000):
    to_keep.add(step)

# Now delete files that are not in the to_keep set
files_to_delete = []
for filename, step in model_files:
    if step not in to_keep:
        files_to_delete.append(filename)

# Print information and confirmation
print(f"Found {len(model_files)} model checkpoint files.")
print(f"Planning to keep {len(to_keep)} files and delete {len(files_to_delete)} files.")
print("\nFiles to be kept:")
for filename, step in model_files:
    if step in to_keep:
        print(f"  {os.path.basename(filename)}")

print("\nFiles to be deleted:")
for filename in files_to_delete:
    print(f"  {os.path.basename(filename)}")

# Safety mechanism - ask for confirmation
confirmation = input("\nDo you want to proceed with deletion? (yes/no): ")
if confirmation.lower() == "yes":
    for filename in files_to_delete:
        try:
            os.remove(filename)
            print(f"Deleted: {os.path.basename(filename)}")
        except Exception as e:
            print(f"Error deleting {os.path.basename(filename)}: {e}")
    print(f"\nDeletion complete. Deleted {len(files_to_delete)} files.")
else:
    print("Deletion cancelled.")
