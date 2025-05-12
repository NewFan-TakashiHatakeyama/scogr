import os
import json
import glob
from collections import Counter, defaultdict

# Define restoration class IDs
RESTORATION_CLASSES = {
    "Inlay": 0,
    "Crown": 1,
    "Treated tooth": 2,
    "Other": 3
}

# Japanese to English mapping
JP_TO_EN_RESTORATION = {
    "インレー": "Inlay",
    "クラウン": "Crown",
    "処置歯": "Treated tooth"
    # Other types will be mapped to "Other"
}

def create_restoration_annotations(base_dir="dataset/Teeth Segmentation on dental X-ray images", output_dir=None):
    """
    Create YOLO format annotations for dental restorations from JSON annotations.
    
    Args:
        base_dir (str): Base directory containing the dataset
        output_dir (str, optional): Output directory for annotations. If None, uses base_dir/restoration_labels
    
    Returns:
        dict: Statistics about the created annotations
    """
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(base_dir, "restoration_labels")
    
    annotation_dir = os.path.join(base_dir, "annotation")
    labels_dir = os.path.join(base_dir, "labels")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics counters
    total_annotations = 0
    stats_by_class = Counter()
    teeth_with_restoration = defaultdict(int)
    images_with_restoration = defaultdict(set)
    
    # Process each annotation file
    for annotation_file in glob.glob(os.path.join(annotation_dir, "dental_*.json")):
        # Extract the image number from the annotation filename
        image_num = annotation_file.split("dental_")[-1].split(".json")[0]
        
        # Find corresponding label file
        label_file = os.path.join(labels_dir, f"{image_num}.txt")
        
        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} not found, skipping...")
            continue
        
        # Read annotation data
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # Read label data (YOLO format bounding boxes)
        with open(label_file, 'r', encoding='utf-8') as f:
            label_lines = f.readlines()
        
        # Create a mapping from tooth_id to bounding box
        tooth_bboxes = {}
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                tooth_id = parts[0]  # Class ID corresponds to tooth number in the classes.txt
                bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                tooth_bboxes[tooth_id] = bbox
        
        # Map classes.txt tooth IDs to FDI notation used in JSON
        # The classes.txt contains IDs like 0, 1, 2 which map to FDI tooth numbers
        # Read the classes mapping
        classes_file = os.path.join(base_dir, "classes.txt")
        class_to_tooth = {}
        with open(classes_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                class_to_tooth[str(idx)] = line.strip()
        
        # Create reverse mapping from FDI tooth number to class ID
        tooth_to_class = {v: k for k, v in class_to_tooth.items()}
        
        # Output file for restoration annotations
        output_file = os.path.join(output_dir, f"{image_num}.txt")
        
        # Keep track of restoration annotations for this image
        image_restorations = []
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Check each tooth for restorations
            for tooth_id, tooth_data in annotation_data.get("teeth", {}).items():
                restoration = tooth_data.get("restoration", {})
                
                # Check if class ID exists for this tooth
                if tooth_id not in tooth_to_class:
                    continue
                
                class_id = tooth_to_class[tooth_id]
                
                # Check if we have bbox for this tooth
                if class_id not in tooth_bboxes:
                    continue
                
                bbox = tooth_bboxes[class_id]
                has_restoration = False
                
                # Check for restoration type
                jp_restoration_type = restoration.get("type")
                
                if jp_restoration_type in JP_TO_EN_RESTORATION:
                    # Map known Japanese terms to English
                    en_restoration_type = JP_TO_EN_RESTORATION[jp_restoration_type]
                    out_f.write(f"{RESTORATION_CLASSES[en_restoration_type]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                    stats_by_class[en_restoration_type] += 1
                    teeth_with_restoration[tooth_id] += 1
                    images_with_restoration[en_restoration_type].add(image_num)
                    total_annotations += 1
                    has_restoration = True
                    image_restorations.append(f"Tooth {tooth_id}: {en_restoration_type} ({jp_restoration_type})")
                elif jp_restoration_type is not None and jp_restoration_type != "null":
                    # Other type of restoration
                    out_f.write(f"{RESTORATION_CLASSES['Other']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                    stats_by_class['Other'] += 1
                    teeth_with_restoration[tooth_id] += 1
                    images_with_restoration['Other'].add(image_num)
                    total_annotations += 1
                    has_restoration = True
                    image_restorations.append(f"Tooth {tooth_id}: Other ({jp_restoration_type})")
        
        # Print details for this image
        if image_restorations:
            print(f"Image {image_num} - Restorations found:")
            for restoration in image_restorations:
                print(f"  - {restoration}")
        else:
            print(f"Image {image_num} - No restorations found")
    
    # Create a README with details about the dataset
    create_restoration_readme(output_dir, RESTORATION_CLASSES, JP_TO_EN_RESTORATION, total_annotations, 
                           stats_by_class, teeth_with_restoration, images_with_restoration)
    
    print(f"\nSummary:")
    print(f"- Generated restoration annotations for {len(glob.glob(os.path.join(output_dir, '*.txt')))} images")
    print(f"- Total annotations: {total_annotations}")
    for restoration, count in stats_by_class.items():
        print(f"- {restoration}: {count} annotations in {len(images_with_restoration[restoration])} images")
    print(f"\nDetailed information has been written to {os.path.join(output_dir, 'README.md')}")
    
    # Return statistics
    return {
        "total_annotations": total_annotations,
        "stats_by_class": stats_by_class,
        "teeth_with_restoration": teeth_with_restoration,
        "images_with_restoration": images_with_restoration,
        "output_dir": output_dir
    }

def create_restoration_readme(output_dir, restoration_classes, jp_to_en_mapping, total_annotations, 
                           stats_by_class, teeth_with_restoration, images_with_restoration):
    """
    Create a README file with dataset statistics for dental restoration annotations.
    
    Args:
        output_dir (str): Directory to write the README to
        restoration_classes (dict): Mapping of restoration types to class IDs
        jp_to_en_mapping (dict): Mapping of Japanese restoration names to English
        total_annotations (int): Total number of annotations created
        stats_by_class (Counter): Count of annotations by class
        teeth_with_restoration (dict): Count of restorations by tooth ID
        images_with_restoration (dict): Set of images containing each restoration type
    """
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as readme_file:
        readme_file.write("# Dental Restoration YOLO Dataset\n\n")
        readme_file.write("## Overview\n")
        readme_file.write("This dataset contains YOLO format annotations for dental restoration detection based on dental X-ray images.\n\n")
        
        readme_file.write("## Classes\n")
        readme_file.write("The following classes are used for restoration annotation:\n\n")
        readme_file.write("| Class ID | Restoration Type | Japanese |\n")
        readme_file.write("|----------|-----------------|----------|\n")
        for restoration, class_id in restoration_classes.items():
            jp_restoration = next((jp for jp, en in jp_to_en_mapping.items() if en == restoration), "-")
            readme_file.write(f"| {class_id} | {restoration} | {jp_restoration} |\n")
        
        readme_file.write("\n## Dataset Statistics\n\n")
        readme_file.write(f"- Total annotations: {total_annotations}\n")
        
        readme_file.write("\n### Annotations by Class\n\n")
        readme_file.write("| Class | Count | Images |\n")
        readme_file.write("|-------|-------|--------|\n")
        for restoration, count in stats_by_class.items():
            readme_file.write(f"| {restoration} | {count} | {len(images_with_restoration[restoration])} |\n")
        
        readme_file.write("\n### Top 10 Teeth with Most Restorations\n\n")
        readme_file.write("| Tooth ID | Restoration Count |\n")
        readme_file.write("|----------|----------------|\n")
        for tooth_id, count in sorted(teeth_with_restoration.items(), key=lambda x: x[1], reverse=True)[:10]:
            readme_file.write(f"| {tooth_id} | {count} |\n") 