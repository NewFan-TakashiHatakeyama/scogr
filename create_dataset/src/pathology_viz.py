import os
import glob
import cv2
import numpy as np

# Define colors for different classes (BGR format for OpenCV)
# Teeth bounding boxes will be white
TEETH_COLOR = (255, 255, 255)
# Pathology colors (BGR format)
PATHOLOGY_COLORS = {
    0: (0, 0, 255),    # Caries - Red
    1: (255, 0, 0),    # Periapical lesion - Blue
    2: (0, 255, 0),    # Root resorption - Green
    3: (128, 0, 128)   # Bone loss - Purple
}

# Class names for the legend
PATHOLOGY_NAMES = {
    0: "Caries",
    1: "Periapical lesion",
    2: "Root resorption",
    3: "Bone loss"
}

# Japanese equivalent (for display)
JP_PATHOLOGY_NAMES = {
    0: "う蝕",
    1: "根尖病変",
    2: "歯根吸収",
    3: "骨吸収"
}

def visualize_pathology_annotations(base_dir="dataset/Teeth Segmentation on dental X-ray images", 
                                   pathology_labels_dir=None, images_dir=None, 
                                   labels_dir=None, output_dir=None):
    """
    Create visualizations of dental pathology annotations.
    
    Args:
        base_dir (str): Base directory containing the dataset
        pathology_labels_dir (str, optional): Directory containing pathology labels
        images_dir (str, optional): Directory containing images
        labels_dir (str, optional): Directory containing original tooth labels
        output_dir (str, optional): Directory to save visualizations
        
    Returns:
        dict: Statistics about the created visualizations
    """
    # Set default directories if not specified
    if pathology_labels_dir is None:
        pathology_labels_dir = os.path.join(base_dir, "pathology_labels")
    if images_dir is None:
        images_dir = os.path.join(base_dir, "images")
    if labels_dir is None:
        labels_dir = os.path.join(base_dir, "labels")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "pathology_visualization")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of visualizations created
    viz_count = 0
    classes_found_total = set()
    
    # Process each image with pathology annotations
    for pathology_label_file in glob.glob(os.path.join(pathology_labels_dir, "*.txt")):
        if os.path.basename(pathology_label_file) == "README.md":
            continue
            
        image_num = os.path.basename(pathology_label_file).split(".")[0]
        
        # Find image file
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(images_dir, f"{image_num}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            print(f"Warning: Image for {image_num} not found, skipping...")
            continue
        
        # Find original labels file
        label_path = os.path.join(labels_dir, f"{image_num}.txt")
        if not os.path.exists(label_path):
            print(f"Warning: Original label file for {image_num} not found, skipping...")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}, skipping...")
            continue
        
        # Create a copy for visualization
        viz_image = image.copy()
        
        # Draw original teeth bounding boxes (white)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    tooth_id = parts[0]
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    # Draw teeth bounding boxes with thinner lines
                    draw_bbox(viz_image, bbox, TEETH_COLOR, thickness=1)
        
        # Draw pathology bounding boxes with colored overlays
        classes_found = set()
        with open(pathology_label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                    # Draw pathology bounding boxes with thicker lines
                    draw_bbox(viz_image, bbox, PATHOLOGY_COLORS[class_id], thickness=2, 
                             text=PATHOLOGY_NAMES[class_id])
                    classes_found.add(class_id)
                    classes_found_total.add(class_id)
        
        # Add legend
        legend_height = 30 * len(classes_found)
        legend = np.ones((legend_height, 300, 3), dtype=np.uint8) * 255
        
        y_offset = 20
        for class_id in sorted(classes_found):
            color = PATHOLOGY_COLORS[class_id]
            en_name = PATHOLOGY_NAMES[class_id]
            jp_name = JP_PATHOLOGY_NAMES[class_id]
            cv2.rectangle(legend, (10, y_offset-15), (30, y_offset+5), color, -1)
            cv2.putText(legend, f"{en_name} ({jp_name})", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 30
        
        # Create output image with legend
        output_height = viz_image.shape[0] + legend.shape[0]
        output_width = max(viz_image.shape[1], legend.shape[1])
        output_image = np.ones((output_height, output_width, 3), dtype=np.uint8) * 255
        
        # Place the visualization image
        output_image[:viz_image.shape[0], :viz_image.shape[1]] = viz_image
        
        # Place the legend below
        legend_y_start = viz_image.shape[0]
        legend_x_start = (output_width - legend.shape[1]) // 2  # Center the legend
        output_image[legend_y_start:legend_y_start+legend.shape[0], 
                    legend_x_start:legend_x_start+legend.shape[1]] = legend
        
        # Save the output
        output_path = os.path.join(output_dir, f"{image_num}_pathology_viz.jpg")
        cv2.imwrite(output_path, output_image)
        print(f"Created visualization for image {image_num}")
        viz_count += 1
    
    print(f"\nVisualization complete! Output saved to {output_dir}")
    print(f"Created {viz_count} visualizations")
    
    # Return statistics
    return {
        "visualization_count": viz_count,
        "classes_found": classes_found_total,
        "output_dir": output_dir
    }

def draw_bbox(image, bbox, color, thickness=2, text=None):
    """
    Draw a bounding box on an image.
    
    Args:
        image: Image to draw on
        bbox: Bounding box in YOLO format [x_center, y_center, width, height]
        color: BGR color tuple
        thickness: Line thickness
        text: Optional text to draw with the box
        
    Returns:
        image: The image with bounding box drawn
    """
    h, w, _ = image.shape
    x_center, y_center, bbox_width, bbox_height = bbox
    
    # Convert from normalized YOLO format to pixel coordinates
    x1 = int((x_center - bbox_width/2) * w)
    y1 = int((y_center - bbox_height/2) * h)
    x2 = int((x_center + bbox_width/2) * w)
    y2 = int((y_center + bbox_height/2) * h)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw text if provided
    if text:
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return image 