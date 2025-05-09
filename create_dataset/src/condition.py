import os
import json
import glob
from collections import Counter, defaultdict

# Define condition class IDs
CONDITION_CLASSES = {
    "Unerupted tooth": 0,       # 未萌歯（Unerupted tooth）
    "Congenitally missing tooth": 1,  # 先欠歯（Congenitally missing tooth）
    "Impacted tooth": 2         # 埋伏歯（Impacted tooth）
}

# Japanese to English mapping
JP_TO_EN_CONDITION = {
    "未萌歯": "Unerupted tooth",
    "先欠歯": "Congenitally missing tooth",
    "埋伏歯": "Impacted tooth"
}

def create_condition_annotations(base_dir="dataset/Teeth Segmentation on dental X-ray images", output_dir=None):
    """
    Create YOLO format annotations for tooth conditions from JSON annotations.
    
    Args:
        base_dir (str): Base directory containing the dataset
        output_dir (str, optional): Output directory for annotations. If None, uses base_dir/condition_labels
    
    Returns:
        dict: Statistics about the created annotations
    """
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(base_dir, "condition_labels")
    
    annotation_dir = os.path.join(base_dir, "annotation")
    labels_dir = os.path.join(base_dir, "labels")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics counters
    total_annotations = 0
    stats_by_class = Counter()
    teeth_with_condition = defaultdict(int)
    images_with_condition = defaultdict(set)
    
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
        
        # Output file for condition annotations
        output_file = os.path.join(output_dir, f"{image_num}.txt")
        
        # Keep track of condition annotations for this image
        image_conditions = []
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Check each tooth for conditions
            for tooth_id, tooth_data in annotation_data.get("teeth", {}).items():
                status = tooth_data.get("status", {})
                
                # Check if class ID exists for this tooth
                if tooth_id not in tooth_to_class:
                    continue
                
                class_id = tooth_to_class[tooth_id]
                
                # Check if we have bbox for this tooth
                if class_id not in tooth_bboxes:
                    continue
                
                bbox = tooth_bboxes[class_id]
                has_condition = False
                
                # Check for condition
                jp_condition = status.get("condition")
                # Skip Existing tooth (残存歯) and only include the other conditions
                if jp_condition in JP_TO_EN_CONDITION:
                    en_condition = JP_TO_EN_CONDITION[jp_condition]
                    out_f.write(f"{CONDITION_CLASSES[en_condition]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                    stats_by_class[en_condition] += 1
                    teeth_with_condition[tooth_id] += 1
                    images_with_condition[en_condition].add(image_num)
                    total_annotations += 1
                    has_condition = True
                    image_conditions.append(f"Tooth {tooth_id}: {en_condition} ({jp_condition})")
        
        # Print details for this image
        if image_conditions:
            print(f"Image {image_num} - Conditions found:")
            for condition in image_conditions:
                print(f"  - {condition}")
        else:
            print(f"Image {image_num} - No conditions found")
    
    # Create a README with details about the dataset
    create_condition_readme(output_dir, CONDITION_CLASSES, JP_TO_EN_CONDITION, total_annotations, 
                           stats_by_class, teeth_with_condition, images_with_condition)
    
    print(f"\nSummary:")
    print(f"- Generated condition annotations for {len(glob.glob(os.path.join(output_dir, '*.txt')))} images")
    print(f"- Total annotations: {total_annotations}")
    for condition, count in stats_by_class.items():
        print(f"- {condition}: {count} annotations in {len(images_with_condition[condition])} images")
    print(f"\nDetailed information has been written to {os.path.join(output_dir, 'README.md')}")
    
    # Return statistics
    return {
        "total_annotations": total_annotations,
        "stats_by_class": stats_by_class,
        "teeth_with_condition": teeth_with_condition,
        "images_with_condition": images_with_condition,
        "output_dir": output_dir
    }

def create_condition_readme(output_dir, condition_classes, jp_to_en_mapping, total_annotations, 
                           stats_by_class, teeth_with_condition, images_with_condition):
    """
    Create a README file with dataset statistics for tooth condition annotations.
    
    Args:
        output_dir (str): Directory to write the README to
        condition_classes (dict): Mapping of condition names to class IDs
        jp_to_en_mapping (dict): Mapping of Japanese condition names to English
        total_annotations (int): Total number of annotations created
        stats_by_class (Counter): Count of annotations by class
        teeth_with_condition (dict): Count of conditions by tooth ID
        images_with_condition (dict): Set of images containing each condition
    """
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as readme_file:
        readme_file.write("# Dental Condition YOLO Dataset\n\n")
        readme_file.write("## Overview\n")
        readme_file.write("This dataset contains YOLO format annotations for dental condition detection (except Existing teeth) based on dental X-ray images.\n\n")
        
        readme_file.write("## Classes\n")
        readme_file.write("The following classes are used for condition annotation:\n\n")
        readme_file.write("| Class ID | Condition Type | Japanese |\n")
        readme_file.write("|----------|---------------|----------|\n")
        for condition, class_id in condition_classes.items():
            jp_condition = next((jp for jp, en in jp_to_en_mapping.items() if en == condition), "-")
            readme_file.write(f"| {class_id} | {condition} | {jp_condition} |\n")
        
        readme_file.write("\n## Dataset Statistics\n\n")
        readme_file.write(f"- Total annotations: {total_annotations}\n")
        
        readme_file.write("\n### Annotations by Class\n\n")
        readme_file.write("| Class | Count | Images |\n")
        readme_file.write("|-------|-------|--------|\n")
        for condition, count in stats_by_class.items():
            readme_file.write(f"| {condition} | {count} | {len(images_with_condition[condition])} |\n")
        
        readme_file.write("\n### Top 10 Teeth with Most Conditions\n\n")
        readme_file.write("| Tooth ID | Condition Count |\n")
        readme_file.write("|----------|----------------|\n")
        for tooth_id, count in sorted(teeth_with_condition.items(), key=lambda x: x[1], reverse=True)[:10]:
            readme_file.write(f"| {tooth_id} | {count} |\n") 