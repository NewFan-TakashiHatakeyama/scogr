import os
import glob
import random
import shutil

def create_restoration_train_val_split(base_dir="dataset/Teeth Segmentation on dental X-ray images", 
                                      restoration_labels_dir=None, images_dir=None, val_ratio=0.2):
    """
    Create train-val split for dental restoration annotations.
    
    Args:
        base_dir (str): Base directory containing the dataset
        restoration_labels_dir (str, optional): Directory containing restoration labels
        images_dir (str, optional): Directory containing images
        val_ratio (float): Ratio of data to use for validation
        
    Returns:
        dict: Statistics about the created train-val split
    """
    # Set default directories if not specified
    if restoration_labels_dir is None:
        restoration_labels_dir = os.path.join(base_dir, "restoration_labels")
    if images_dir is None:
        images_dir = os.path.join(base_dir, "images")
    
    # Find all the restoration label files
    restoration_files = glob.glob(os.path.join(restoration_labels_dir, "*.txt"))
    # Exclude README.md
    restoration_files = [f for f in restoration_files if os.path.basename(f) != "README.md"]
    
    # Check if files have content (not empty)
    valid_files = []
    for file_path in restoration_files:
        if os.path.getsize(file_path) > 0:
            valid_files.append(file_path)
    
    # Determine the split ratio
    random.shuffle(valid_files)
    
    # Calculate split indices
    num_val = max(1, int(len(valid_files) * val_ratio))
    val_files = valid_files[:num_val]
    train_files = valid_files[num_val:]
    
    # Generate train.txt and val.txt with paths to images
    train_file = os.path.join(restoration_labels_dir, "train.txt")
    val_file = os.path.join(restoration_labels_dir, "val.txt")
    
    with open(train_file, 'w') as f:
        for label_file in train_files:
            image_path = verify_image_exists(label_file, images_dir)
            if image_path:
                # Use relative path for compatibility
                rel_path = os.path.relpath(image_path, os.path.dirname(restoration_labels_dir))
                f.write(f"{rel_path}\n")
    
    with open(val_file, 'w') as f:
        for label_file in val_files:
            image_path = verify_image_exists(label_file, images_dir)
            if image_path:
                # Use relative path for compatibility
                rel_path = os.path.relpath(image_path, os.path.dirname(restoration_labels_dir))
                f.write(f"{rel_path}\n")
    
    print(f"Created train/val split:")
    print(f"- Train set: {len(train_files)} files")
    print(f"- Validation set: {len(val_files)} files")
    print(f"Files saved to {train_file} and {val_file}")
    
    # Create dataset directory structure for YOLO format
    dataset_path = os.path.join(base_dir, "restoration_yolo_dataset")
    os.makedirs(os.path.join(dataset_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labels", "val"), exist_ok=True)
    
    # Copy train and validation files
    copy_files(train_files, "train", dataset_path, images_dir)
    copy_files(val_files, "val", dataset_path, images_dir)
    
    # Create YOLO config file in the dataset root
    yaml_content = f"""# Dental Restoration Dataset
path: .
train: images/train
val: images/val

nc: 4
names: ["Inlay", "Crown", "Treated tooth", "Other"]
"""
    
    yaml_path = os.path.join(dataset_path, "restoration.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated YOLO dataset structure at {dataset_path}")
    print("You can train a YOLO model with this dataset using:")
    print(f"yolo task=detect mode=train data={yaml_path}")
    
    # Return statistics
    return {
        "train_count": len(train_files),
        "val_count": len(val_files),
        "train_file": train_file,
        "val_file": val_file,
        "yaml_path": yaml_path,
        "dataset_path": dataset_path
    }

def verify_image_exists(label_file, images_dir):
    """
    Verify if an image exists for a given label file.
    
    Args:
        label_file (str): Path to a label file
        images_dir (str): Directory containing images
        
    Returns:
        str or None: Path to the image if it exists, None otherwise
    """
    image_num = os.path.basename(label_file).split(".")[0]
    for ext in ['.jpg', '.jpeg', '.png']:
        image_path = os.path.join(images_dir, f"{image_num}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None

def copy_files(label_files, target_split, dataset_path, images_dir):
    """
    Copy label and image files to the YOLO dataset structure.
    
    Args:
        label_files (list): List of label files to copy
        target_split (str): Target split ('train' or 'val')
        dataset_path (str): Path to the YOLO dataset directory
        images_dir (str): Directory containing images
    """
    for label_file in label_files:
        # Get image number
        image_num = os.path.basename(label_file).split(".")[0]
        
        # Find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(images_dir, f"{image_num}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            continue
            
        # Copy image
        image_ext = os.path.splitext(image_path)[1]
        dest_image = os.path.join(dataset_path, "images", target_split, f"{image_num}{image_ext}")
        shutil.copy2(image_path, dest_image)
        
        # Copy label
        dest_label = os.path.join(dataset_path, "labels", target_split, f"{image_num}.txt")
        shutil.copy2(label_file, dest_label) 