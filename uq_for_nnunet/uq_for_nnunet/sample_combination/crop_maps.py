#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Description:    Crop uncertainty samples based on the ROI mask to reduce memory 
                usage and speed up computation.

Usage:
    python crop_maps.py --folder /path/to/npz_files \


Arguments:
    --folder        Folder containing .npz prediction files
    --patients      List of patient IDs to process (optional)

===============================================================================
"""

import os
import argparse
import logging
import re
from tqdm import tqdm
import nibabel as nib
import numpy as np

def get_midpoint_of_segmentation(segmentation_arr):
    """
    Get the midpoint of the segmentation mask along the z-axis.

    Args:
        segmentation_arr (np.ndarray): 3D array of the segmentation mask.
    
    Returns:
        midpoint list: Midpoint x, y, z coordinates of the segmentation mask.
    
    """
    # Find the bounding box of the segmentation
    coords = np.where(segmentation_arr > 0)

    # Calculate the midpoint along the z-axis
    z_min, z_max = np.min(coords[0]), np.max(coords[0])
    z_midpoint = (z_min + z_max) // 2

    # Calculate the midpoint along the x and y axes
    x_midpoint = (np.min(coords[1]) + np.max(coords[1])) // 2
    y_midpoint = (np.min(coords[2]) + np.max(coords[2])) // 2

    return [x_midpoint, y_midpoint, z_midpoint]

def crop_maps(folder, patients, methods, cropsize):
    for patient in tqdm(patients, desc="Patients"):
        print(f'evaluating patient {patient}')
        for method in methods:
            if method=="mc_dropout":
                pattern = re.compile(f'.*_{patient}_(0?[1-9]|1[0-9]|20)_.npz')
                all_files = [f for f in os.listdir(folder) if pattern.match(f)]
            elif method=="deep_ensemble":
                all_files = []
                for subfolder in os.listdir(folder):
                    if 'fold' in subfolder:
                        print(subfolder)
                        pattern = re.compile(f'.*_{patient}.npz')
                        subfolder_path = os.path.join(folder, subfolder)
                        patient_file = [f for f in os.listdir(subfolder_path) if pattern.match(f)]
                        print(f'Added {os.path.join(subfolder_path, patient_file[0])} \n')
                        all_files.append(os.path.join(subfolder_path, patient_file[0]))
            elif method=="tta":
                all_files = []
                for subfolder in os.listdir(folder):
                    if 'fold' in subfolder:
                        print(subfolder)
                        pattern = re.compile(f'.*_{patient}.npz')
                        subfolder_path = os.path.join(folder, subfolder)
                        patient_file = [f for f in os.listdir(subfolder_path) if pattern.match(f)]
                        print(f'Added {os.path.join(subfolder_path, patient_file[0])} \n')
                        all_files.append(os.path.join(subfolder_path, patient_file[0]))
            else:
                pattern = re.compile(f'.*_{patient}_.npz')
                all_files = [f for f in os.listdir(folder) if pattern.match(f)]

            print(all_files)

            #load segmentation
            segmentation_file = re.compile(f'.*_{patient}.nii.gz')
            segmentation_arr = nib.load(os.path.join(folder, segmentation_file)).get_fdata()

            print(f"Shape of segmentation array: {segmentation_arr.shape}")
            

            # binarise segmentation
            segmentation_arr[segmentation_arr > 0] = 1

            # get the midpoint of the segmentation
            midpoint = get_midpoint_of_segmentation(segmentation_arr)

            # calculate the crop boundaries based on the midpoint and the specified crop size
            x_size, y_size, z_size = map(int, cropsize)
            x_min = max(midpoint[0] - x_size // 2, 0)
            x_max = min(midpoint[0] + x_size // 2, segmentation_arr.shape[1])
            y_min = max(midpoint[1] - y_size // 2, 0)
            y_max = min(midpoint[1] + y_size // 2, segmentation_arr.shape[2])
            z_min = max(midpoint[2] - z_size // 2, 0)
            z_max = min(midpoint[2] + z_size // 2, segmentation_arr.shape[0])

            print(f"Cropped shape will be from z: {z_min} to {z_max}, x: {x_min} to {x_max}, y: {y_min} to {y_max}")

            # crop the sample maps based on the midpoint and a predefined size
            for file in tqdm(all_files, desc=f"Cropping maps for patient {patient} using method {method}"):
                file_path = os.path.join(folder, file)
                data = np.load(file_path)
                cropped_data = {}
                for key in data:
                    cropped_data[key] = data[key][z_min:z_max, x_min:x_max, y_min:y_max]
                
                # Save the cropped maps to a new .npz file
                cropped_file_path = os.path.join(folder, file.replace('.npz', '_cropped.npz')) #NOTE the old file is overwritten, if you want to keep the original file, change the name here
                np.savez(cropped_file_path, **cropped_data)              
 


def main():
    logging.basicConfig(level=logging.INFO)
    print(" Starting uncertainty map computation script")
    parser = argparse.ArgumentParser(description="Compute uncertainty maps from prediction npz files.")
    parser.add_argument("--folder", type=str, required=True, help="Folder with npz prediction files")
    parser.add_argument("--patients", type=str, nargs='+', default=None, help="List of patient IDs to process. If not provided, all patients in the folder will be processed.")
    parser.add_argument("--methods", nargs="+", required=True, help="List of uncertainty methods used to obtain the samples (mc_dropout, deep_ensemble, tta)") 
    parser.add_argument("--cropsize", nargs=3, type=list, required=True, help="Crop size as three integers: x_size y_size z_size")

    args = parser.parse_args()

   # default process all patients if not specified
    if args.patients is None:
        args.patients = [f.split('_')[1] for f in os.listdir(args.folder) if f.endswith('.npz')]
        logging.info(f"No specific patients provided. Processing all patients: {args.patients}")

    
    crop_maps(folder=args.folder, patients=args.patients, methods=args.methods, cropsize=args.cropsize)

if __name__ == "__main__":
    main()
