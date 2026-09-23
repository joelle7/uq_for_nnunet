#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Description:    Crop uncertainty samples based on the ROI mask to reduce memory
                usage and speed up computation.
                NOTE: this function assumes that the uncertainty maps are in the shape (classes, z, x, y) and the segmentation mask is in the shape (x, y, z).
                And it assumes that the segmentation file and the uncertainty maps are in the same folder and have the same patient ID in their filenames.

Usage:
    python crop_uncertainty_map.py --folder /path/to/npz_files \
                                    --patients patient1 patient2 \
                                    --crop_size 128 128 64

Arguments:
    --folder        Folder containing .npz prediction files
    --patients      List of patient IDs to process (optional)
    --crop_size     Size to crop the maps (x_size, y_size, z_size)

===============================================================================
"""


import argparse
import os
import re
from tqdm import tqdm
import nibabel as nib
import numpy as np

def get_midpoint_of_segmentation(segmentation_arr):
    """
    Get the midpoint of the segmentation mask along the z-axis.
    NOTE: assumes that the segmentation mask is in the shape (x, y, z).

    Args:
        segmentation_arr (np.ndarray): 3D array of the segmentation mask.

    Returns:
        midpoint list: Midpoint x, y, z coordinates of the segmentation mask.
    """

    # Find the bounding box of the segmentation
    coords = np.where(segmentation_arr > 0)

    # Calculate the midpoint along the z-axis
    z_min, z_max = np.min(coords[2]), np.max(coords[2])
    z_midpoint = (z_min + z_max) // 2

    # Calculate the midpoint along the x and y axes
    x_midpoint = (np.min(coords[0]) + np.max(coords[0])) // 2
    y_midpoint = (np.min(coords[1]) + np.max(coords[1])) // 2

    print(f'Midpoint of segmentation: x={x_midpoint}, y={y_midpoint}, z={z_midpoint}')
    return [x_midpoint, y_midpoint, z_midpoint]

def crop_single_uncertainty_map(uncertainty_map, segmentation_arr, crop_size):
    """
    Crop a single uncertainty map based on the segmentation midpoint and crop size.

    Args:
        uncertainty_map (np.ndarray): 4D array of the uncertainty map (classes, z, y, x).
        segmentation_midpoint (list): Midpoint x, y, z coordinates of the segmentation mask.
        crop_size (list): Size to crop the maps (x_size, y_size, z_size).

    Returns:
        cropped_map (np.ndarray): Cropped uncertainty map.
    """
    print(f'Segmentation shape: {segmentation_arr.shape}')
    
    segmentation_midpoint = get_midpoint_of_segmentation(segmentation_arr)

    # uncertainty map is in shape classes, z, x, y
    print(f'Original uncertainty map shape: {uncertainty_map.shape}')

    xmin = segmentation_midpoint[0] - crop_size[0] // 2
    xmax = segmentation_midpoint[0] + crop_size[0] // 2
    ymin = segmentation_midpoint[1] - crop_size[1] // 2
    ymax = segmentation_midpoint[1] + crop_size[1] // 2
    zmin = segmentation_midpoint[2] - crop_size[2] // 2
    zmax = segmentation_midpoint[2] + crop_size[2] // 2

    print(f'Cropping coordinates: x({xmin}:{xmax}), y({ymin}:{ymax}), z({zmin}:{zmax})')
    cropped_map = uncertainty_map[
        :,
        zmin:zmax,
        ymin:ymax,
        xmin:xmax
    ]  # crop uncertainty map to [classes, zlim, ylim, xlim] based on the segmentation midpoint and crop size
    print(f'Cropped map shape: {cropped_map.shape}')

    return cropped_map
    



def crop_uncertainty_maps(folder, patients, methods, crop_size): 
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

        segmentation_filename_pattern = re.compile(f'.*_{patient}.nii.gz')
        segmentation_filename = next((f for f in os.listdir(folder) if segmentation_filename_pattern.match(f)), None) # get first matching segmentation file

        print(segmentation_filename)
        segmentation_arr = nib.load(os.path.join(folder, segmentation_filename)).get_fdata()

        # Use the segmentation midpoint to crop the uncertainty maps
        for file in all_files:
            uncertainty_map = np.load(os.path.join(folder, file))['probabilities']

            cropped_map = crop_single_uncertainty_map(uncertainty_map, segmentation_arr, crop_size)
            
            # save as ['probabilities'] to be consistent with the original npz files
            np.savez_compressed(os.path.join(folder, file.replace('.npz', '_cropped.npz')), probabilities=cropped_map)

def main():
    parser = argparse.ArgumentParser(description="Crop uncertainty maps based on the ROI mask.")
    parser.add_argument("--folder", type=str, required=True, help="Folder with npz prediction files")
    parser.add_argument("--patients", type=str, nargs='+', default=None, help="List of patient IDs to process. If not provided, all patients in the folder will be processed.")
    parser.add_argument("--methods", nargs="+", required=True, help="List of uncertainty methods used to obtain the samples (mc_dropout, deep_ensemble, tta)")
    parser.add_argument("--crop_size", nargs='+', type=int, required=True, help="Crop size as three integers: x_size y_size z_size")

    args = parser.parse_args()

    if args.patients is None:
        # If no specific patients are provided, process all patients in the folder
        args.patients = [f.split('_')[1] for f in os.listdir(args.folder) if f.endswith('.npz')]
        args.patients = list(set(args.patients))  # Remove duplicates
        print(f"No specific patients provided. Processing all patients: {args.patients}")

    crop_uncertainty_maps(folder=args.folder, patients=args.patients, methods=args.methods, crop_size=args.crop_size)