import os
import json
import glob
from collections import Counter, defaultdict

# Define pathology class IDs
PATHOLOGY_CLASSES = {
    "Caries": 0,           # う蝕（Caries）
    "Periapical lesion": 1,  # 根尖病変（Periapical lesion）
    "Root resorption": 2,    # 歯根吸収（Root resorption）
    "Bone loss": 3          # 骨吸収（Bone loss）
}

# Japanese to English mapping
JP_TO_EN_PATHOLOGY = {
    "う蝕": "Caries",
    "根尖病変": "Periapical lesion",
    "歯根吸収": "Root resorption",
    "骨吸収": "Bone loss"
}

def create_pathology_annotations(base_dir="dataset/Teeth Segmentation on dental X-ray images", output_dir=None):
    """
    Create YOLO format annotations for dental pathologies from JSON annotations.
    
    Args:
        base_dir (str): Base directory containing the dataset
        output_dir (str, optional): Output directory for annotations. If None, uses base_dir/pathology_labels
    
    Returns:
        dict: Statistics about the created annotations
    """
    # Set default output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(base_dir, "pathology_labels")
    
    annotation_dir = os.path.join(base_dir, "annotation")
    labels_dir = os.path.join(base_dir, "labels")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics counters
    total_annotations = 0
    stats_by_class = Counter()
    teeth_with_pathology = defaultdict(int)
    images_with_pathology = defaultdict(set)
    
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
        # Read the classes mapping
        classes_file = os.path.join(base_dir, "classes.txt")
        class_to_tooth = {}
        with open(classes_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                class_to_tooth[str(idx)] = line.strip()
        
        # Create reverse mapping from FDI tooth number to class ID
        tooth_to_class = {v: k for k, v in class_to_tooth.items()}
        
        # Output file for pathology annotations
        output_file = os.path.join(output_dir, f"{image_num}.txt")
        
        # Keep track of pathology annotations for this image
        image_pathologies = []
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Check each tooth for pathologies
            for tooth_id, tooth_data in annotation_data.get("teeth", {}).items():
                pathology = tooth_data.get("pathology", {})
                
                # Check if class ID exists for this tooth
                if tooth_id not in tooth_to_class:
                    continue
                
                class_id = tooth_to_class[tooth_id]
                
                # Check if we have bbox for this tooth
                if class_id not in tooth_bboxes:
                    continue
                
                bbox = tooth_bboxes[class_id]
                has_pathology = False
                
                # Check for pathology type
                jp_pathology = pathology.get("type")
                if jp_pathology in JP_TO_EN_PATHOLOGY:
                    en_pathology = JP_TO_EN_PATHOLOGY[jp_pathology]
                    out_f.write(f"{PATHOLOGY_CLASSES[en_pathology]} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                    stats_by_class[en_pathology] += 1
                    teeth_with_pathology[tooth_id] += 1
                    images_with_pathology[en_pathology].add(image_num)
                    total_annotations += 1
                    has_pathology = True
                    image_pathologies.append(f"Tooth {tooth_id}: {en_pathology} ({jp_pathology})")
        
        # Print details for this image
        if image_pathologies:
            print(f"Image {image_num} - Pathologies found:")
            for pathology in image_pathologies:
                print(f"  - {pathology}")
        else:
            print(f"Image {image_num} - No pathologies found")
    
    # Create a README with details about the dataset
    create_pathology_readme(output_dir, PATHOLOGY_CLASSES, JP_TO_EN_PATHOLOGY, total_annotations, 
                           stats_by_class, teeth_with_pathology, images_with_pathology)
    
    print(f"\nSummary:")
    print(f"- Generated pathology annotations for {len(glob.glob(os.path.join(output_dir, '*.txt')))} images")
    print(f"- Total annotations: {total_annotations}")
    for pathology, count in stats_by_class.items():
        print(f"- {pathology}: {count} annotations in {len(images_with_pathology[pathology])} images")
    print(f"\nDetailed information has been written to {os.path.join(output_dir, 'README.md')}")
    
    # Return statistics
    return {
        "total_annotations": total_annotations,
        "stats_by_class": stats_by_class,
        "teeth_with_pathology": teeth_with_pathology,
        "images_with_pathology": images_with_pathology,
        "output_dir": output_dir
    }

def create_pathology_readme(output_dir, pathology_classes, jp_to_en_mapping, total_annotations, 
                           stats_by_class, teeth_with_pathology, images_with_pathology):
    """
    Create a README file with dataset statistics for dental pathology annotations.
    
    Args:
        output_dir (str): Directory to write the README to
        pathology_classes (dict): Mapping of pathology types to class IDs
        jp_to_en_mapping (dict): Mapping of Japanese pathology names to English
        total_annotations (int): Total number of annotations created
        stats_by_class (Counter): Count of annotations by class
        teeth_with_pathology (dict): Count of pathologies by tooth ID
        images_with_pathology (dict): Set of images containing each pathology type
    """
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as readme_file:
        readme_file.write("# Dental Pathology YOLO Dataset\n\n")
        readme_file.write("## Overview\n")
        readme_file.write("This dataset contains YOLO format annotations for dental pathology detection based on dental X-ray images.\n\n")
        
        readme_file.write("## Classes\n")
        readme_file.write("The following classes are used for pathology annotation:\n\n")
        readme_file.write("| Class ID | Pathology Type | Japanese |\n")
        readme_file.write("|----------|---------------|----------|\n")
        for pathology, class_id in pathology_classes.items():
            jp_pathology = next((jp for jp, en in jp_to_en_mapping.items() if en == pathology), "-")
            readme_file.write(f"| {class_id} | {pathology} | {jp_pathology} |\n")
        
        readme_file.write("\n## Dataset Statistics\n\n")
        readme_file.write(f"- Total annotations: {total_annotations}\n")
        
        readme_file.write("\n### Annotations by Class\n\n")
        readme_file.write("| Class | Count | Images |\n")
        readme_file.write("|-------|-------|--------|\n")
        for pathology, count in stats_by_class.items():
            readme_file.write(f"| {pathology} | {count} | {len(images_with_pathology[pathology])} |\n")
        
        readme_file.write("\n### Top 10 Teeth with Most Pathologies\n\n")
        readme_file.write("| Tooth ID | Pathology Count |\n")
        readme_file.write("|----------|----------------|\n")
        for tooth_id, count in sorted(teeth_with_pathology.items(), key=lambda x: x[1], reverse=True)[:10]:
            readme_file.write(f"| {tooth_id} | {count} |\n") 