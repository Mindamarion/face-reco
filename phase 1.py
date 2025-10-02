import os
import shutil

# === Phase 1: Dataset Creation ===

# Define raw image folders (where your original images are stored)
marion_train = r"C:\Users\LE\OneDrive\mishelly\Desktop\project\marion train"
marion_test  = r"C:\Users\LE\OneDrive\mishelly\Desktop\project\marion test"
mishelly_train = r"C:\Users\LE\OneDrive\mishelly\Desktop\project\mishelly train"
mishelly_test  = r"C:\Users\LE\OneDrive\mishelly\Desktop\project\mishelly test"

# Define dataset output structure
dataset_dir = "dataset"
folders = [
    os.path.join(dataset_dir, "train", "marion"),
    os.path.join(dataset_dir, "test", "marion"),
    os.path.join(dataset_dir, "train", "mishelly"),
    os.path.join(dataset_dir, "test", "mishelly")
]

# Create dataset directories if not exist
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Function to copy images from raw folder → dataset folder
def copy_images(src_folder, dst_folder):
    if not os.path.exists(src_folder):
        print(f"❌ Folder not found: {src_folder}")
        return
    files = os.listdir(src_folder)
    if not files:
        print(f"⚠️ No files found in {src_folder}")
        return
    for img in files:
        src = os.path.join(src_folder, img)
        dst = os.path.join(dst_folder, img)
        if os.path.isfile(src):
            shutil.copy(src, dst)
            print(f"✅ Copied {img} → {dst_folder}")
        else:
            print(f"Skipping non-file: {src}")

# Copy images for each person
copy_images(marion_train, os.path.join(dataset_dir, "train", "marion"))
copy_images(marion_test,  os.path.join(dataset_dir, "test", "marion"))
copy_images(mishelly_train, os.path.join(dataset_dir, "train", "mishelly"))
copy_images(mishelly_test,  os.path.join(dataset_dir, "test", "mishelly"))

print("\n🎉 Phase 1 complete: Dataset structure created successfully!")
