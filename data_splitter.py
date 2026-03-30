import os
import shutil

print("🗂️ Initializing NTU-RGBD Cross-Subject Splitter (with Corruption Filter)...")

# ==========================================
# 1. DIRECTORY SETUP
# ==========================================
RAW_DATA_DIR = "data/raw_skeletons/nturgbd_skeletons_s001_to_s017"
TRAIN_DIR = "data/train_skeletons"
VAL_DIR = "data/val_skeletons"
MISSING_SKELETONS_TXT = "NTU_RGBD_samples_with_missing_skeletons.txt"

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ==========================================
# 2. BUILD THE BLACKLIST
# ==========================================
ignored_samples = set()
try:
    with open(MISSING_SKELETONS_TXT, 'r') as f:
        for line in f:
            # Strip whitespace/newlines just in case
            clean_name = line.strip()
            if clean_name:
                ignored_samples.add(clean_name)
    print(f"🛡️ Loaded {len(ignored_samples)} corrupted filenames to ignore.")
except FileNotFoundError:
    print(f"⚠️ Warning: '{MISSING_SKELETONS_TXT}' not found! No files will be skipped.")

# ==========================================
# 3. THE SORTING ENGINE
# ==========================================
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]

total_files = 0
train_count = 0
val_count = 0
skipped_count = 0

print("⏳ Sorting files and filtering corruption... ")

for filename in os.listdir(RAW_DATA_DIR):
    if not filename.endswith(".skeleton"):
        continue
        
    # The text file doesn't have '.skeleton' in the names, so we strip it to check
    sample_name = filename.replace('.skeleton', '')
    
    # Check the blacklist!
    if sample_name in ignored_samples:
        skipped_count += 1
        continue
        
    total_files += 1
    
    # Extract Person ID (e.g., S001C001P001R001A060 -> P001 -> 1)
    person_id_str = filename.split('P')[1][:3]
    person_id = int(person_id_str)
    
    source_path = os.path.join(RAW_DATA_DIR, filename)
    
    if person_id in training_subjects:
        dest_path = os.path.join(TRAIN_DIR, filename)
        train_count += 1
    else:
        dest_path = os.path.join(VAL_DIR, filename)
        val_count += 1
        
    shutil.move(source_path, dest_path)

print("-" * 50)
print("✅ DATA SPLIT & FILTER COMPLETE")
print(f"Total Clean Skeletons Processed: {total_files}")
print(f"Corrupted Files Skipped:         {skipped_count}")
print(f"Moved to Training Set:           {train_count}")
print(f"Moved to Validation Set:         {val_count}")
print("-" * 50)