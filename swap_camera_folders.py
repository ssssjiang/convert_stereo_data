#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to swap the contents of camera0 and camera1 folders in camera directory structures.
"""

import os
import shutil
import glob
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def swap_camera_folders(base_dir, dry_run=False):
    """
    Swap the contents of camera0 and camera1 folders in all camera directories.
    
    Args:
        base_dir (str): The base directory to search for camera folders
        dry_run (bool): If True, only print what would be done without making changes
    
    Returns:
        int: Number of camera pairs processed
    """
    # Find all possible camera paths
    search_pattern = os.path.join(base_dir, "**/camera")
    camera_dirs = glob.glob(search_pattern, recursive=True)
    
    if not camera_dirs:
        logger.warning(f"No camera directories found in {base_dir}")
        return 0
    
    logger.info(f"Found {len(camera_dirs)} camera directories")
    processed_count = 0
    
    for camera_dir in camera_dirs:
        camera0_path = os.path.join(camera_dir, "camera0")
        camera1_path = os.path.join(camera_dir, "camera1")
        
        # Check if both camera folders exist
        if not os.path.exists(camera0_path) or not os.path.exists(camera1_path):
            logger.warning(f"Missing camera0 or camera1 in {camera_dir}, skipping...")
            continue
        
        logger.info(f"Processing: {camera_dir}")
        
        if dry_run:
            logger.info(f"Would swap {camera0_path} and {camera1_path}")
            processed_count += 1
            continue
        
        # Create temp directory
        temp_path = os.path.join(camera_dir, "camera_temp")
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        
        try:
            # Move camera0 to temp
            logger.info(f"Moving {camera0_path} to temporary folder")
            shutil.move(camera0_path, temp_path)
            
            # Move camera1 to camera0
            logger.info(f"Moving {camera1_path} to {camera0_path}")
            shutil.move(camera1_path, camera0_path)
            
            # Move temp to camera1
            logger.info(f"Moving temporary folder to {camera1_path}")
            shutil.move(temp_path, camera1_path)
            
            processed_count += 1
            logger.info(f"Successfully swapped camera0 and camera1 in {camera_dir}")
            
        except Exception as e:
            logger.error(f"Error swapping folders in {camera_dir}: {e}")
            # Try to recover if possible
            if os.path.exists(temp_path) and not os.path.exists(camera0_path):
                logger.info("Attempting recovery...")
                try:
                    shutil.move(temp_path, camera0_path)
                    logger.info("Recovered camera0 folder")
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Swap camera0 and camera1 folders in camera directories")
    parser.add_argument("base_dir", help="Base directory to search for camera folders")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done without making changes")
    
    args = parser.parse_args()
    
    # Convert to absolute path and verify it exists
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        logger.error(f"Directory does not exist: {base_dir}")
        return 1
    
    logger.info(f"Starting camera folder swap in {base_dir}")
    if args.dry_run:
        logger.info("DRY RUN MODE: No files will be modified")
    
    processed_count = swap_camera_folders(base_dir, args.dry_run)
    
    if processed_count > 0:
        logger.info(f"Successfully processed {processed_count} camera pairs")
        return 0
    else:
        logger.warning("No camera folders were processed")
        return 1

if __name__ == "__main__":
    exit(main()) 