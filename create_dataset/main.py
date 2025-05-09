#!/usr/bin/env python
"""
Main script for creating and visualizing YOLO format annotations for dental X-ray images.
"""

import argparse
import sys
import os

# Import modules
from src.condition import create_condition_annotations
from src.condition_viz import visualize_condition_annotations
from src.condition_split import create_condition_train_val_split
from src.restoration import create_restoration_annotations
from src.restoration_viz import visualize_restoration_annotations
from src.restoration_split import create_restoration_train_val_split
from src.pathology import create_pathology_annotations
from src.pathology_viz import visualize_pathology_annotations
from src.pathology_split import create_pathology_train_val_split

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create and visualize YOLO format annotations for dental X-ray images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--type', default='restoration',
                       help='Type of dataset to process condition restoration pathology')
    
    # Optional arguments
    parser.add_argument('--base-dir', 
                       default='dataset/Teeth Segmentation on dental X-ray images',
                       help='Base directory containing the dataset')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of data to use for validation')
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        if args.type == 'condition':
            print("\n======= CREATING CONDITION ANNOTATIONS =======\n")
            stats = create_condition_annotations(base_dir=args.base_dir)
            
            print("\n======= CREATING CONDITION VISUALIZATIONS =======\n")
            visualize_condition_annotations(base_dir=args.base_dir)
            
            print("\n======= CREATING CONDITION TRAIN-VAL SPLIT =======\n")
            create_condition_train_val_split(
                base_dir=args.base_dir,
                val_ratio=args.val_ratio
            )
        
        elif args.type == 'restoration':
            print("\n======= CREATING RESTORATION ANNOTATIONS =======\n")
            stats = create_restoration_annotations(base_dir=args.base_dir)
            
            print("\n======= CREATING RESTORATION VISUALIZATIONS =======\n")
            visualize_restoration_annotations(base_dir=args.base_dir)
            
            print("\n======= CREATING RESTORATION TRAIN-VAL SPLIT =======\n")
            create_restoration_train_val_split(
                base_dir=args.base_dir,
                val_ratio=args.val_ratio
            )
        
        elif args.type == 'pathology':
            print("\n======= CREATING PATHOLOGY ANNOTATIONS =======\n")
            stats = create_pathology_annotations(base_dir=args.base_dir)
            
            print("\n======= CREATING PATHOLOGY VISUALIZATIONS =======\n")
            visualize_pathology_annotations(base_dir=args.base_dir)
            
            print("\n======= CREATING PATHOLOGY TRAIN-VAL SPLIT =======\n")
            create_pathology_train_val_split(
                base_dir=args.base_dir,
                val_ratio=args.val_ratio
            )
        
        return 0
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 