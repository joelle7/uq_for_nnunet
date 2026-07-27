#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Description:    Combines multiple segmentation predictions (e.g., MC Dropout,
                Ensemble, or TTA) into a single final segmentation per patient.

Usage:
    python combine_segmentation.py --method mc_dropout \
                                   --folder /path/to/predictions \
                                   --patients 001 002 003 \
                                   --output_dir /path/to/save/combined

TO DO: logic of this file can be improved, a lot.
===============================================================================
"""

import os
import re
import nibabel as nib
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

def setup_logging(output_dir: str) -> str:
    """
    Set up logging to print to console and save to a timestamped text file.

    Args:
        output_dir (str): Directory where the log file will be saved.

    Returns:
        str: Path to the log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"log_combine_segmentation_{timestamp}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging started. Log file: {log_file}")
    return log_file

def combine_mc_segmentation(folder: str, patients: list, output_dir: str):
    """
    Combine multiple MC dropout segmentations for each patient into a final segmentation.

    Args:
        folder (str): Folder containing MC segmentation .nii.gz files.
        patients (list): List of patient IDs to process.
        output_dir (str): Directory to save combined segmentations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_patients = len(patients)
    print(f'Evaluating patient\n {patients} \n\n')
    for p_idx, patient in enumerate(patients, start=1):
        tqdm.write(f"Processing patient {p_idx} / {total_patients} (ID: {patient})")
        
        # Regex to match all prediction files for this patient
        pattern = re.compile(f'.*_{patient}_(0?[1-9]|1[0-9]|20)_.nii.gz')
        all_seg = []

        # Collect all matching files
        all_files = [f for f in os.listdir(folder) if pattern.match(f)]
        total_files = len(all_files)
        if total_files == 0:
            logging.warning(f"No files found for patient {patient} in {folder}")
            continue

        for f_idx, filename in enumerate(tqdm(all_files, desc="Samples", unit="file"), start=1):
            logging.info(f"[{f_idx}/{total_files}] Loading file: {filename}")
            seg = nib.load(os.path.join(folder, filename)).get_fdata()
            all_seg.append(seg)

        all_seg = np.array(all_seg)
        shape = all_seg.shape[1:]  # CT volume shape
        combined_seg = np.zeros(shape, dtype=np.int32)

        logging.info(f"Combining {len(all_seg)} samples for patient {patient} using majority vote")
        for slice_idx in tqdm(range(shape[0]), desc=f"Slices patient {patient}", unit="slice"):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    combined_seg[slice_idx, j, k] = np.bincount(all_seg[:, slice_idx, j, k].astype(int)).argmax()

        # Save the combined segmentation as NIfTI
        combined_nii = nib.Nifti1Image(combined_seg, np.eye(4))
        output_file = os.path.join(output_dir, f"combined_segmentation_patient_{patient}_MCD.nii.gz")
        nib.save(combined_nii, output_file)
        logging.info(f"Saved combined segmentation for patient {patient} at {output_file}")

def combine_de_segmentation(folder: str, patients: list, output_dir: str):
    # NOTE: this function is not checked
    pattern = re.compile(f'*_{patient}.nii.gz')
    for patient in patients:
        logging.info(f"Processing patient {patient}")
        all_seg = []

        for fold in folds:
            deep_ensemble_folder = os.path.join(deep_ensemble_main, fold)
            for filename in os.listdir(deep_ensemble_folder):
                match = pattern.match(filename)
                if match:
                    logging.info(f"Loading file: {filename}")
                    seg = nib.load(os.path.join(deep_ensemble_folder, filename)).get_fdata()
                    all_seg.append(seg)            
        
        shape = np.shape(all_seg)[1:] #skip first channels, this is the number of folds
        all_seg = np.array(all_seg)
        avg_pred_seg = np.zeros(shape)
        
        logging.info(f"Combining {len(all_seg)} samples for patient {patient} using majority vote")
        
        #determine most occuring ROI in each voxel
        for i in tqdm(range(shape[0])):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    vals = all_seg[:,i,j,k]
                    avg_pred_seg[i,j,k] = np.bincount(vals.astype(int)).argmax()
        
        # Save the combined segmentation as NIfTI
        combined_nii = nib.Nifti1Image(avg_pred_seg, np.eye(4))
        output_file = os.path.join(output_dir, f"combined_segmentation_patient_{patient}_DE.nii.gz")
        nib.save(combined_nii, output_file) 
        logging.info(f"Saved combined segmentation for patient {patient} at {output_file}")

def combine_tta_segmentation(folder: str, patients: list, output_dir: str):
    # NOTE: this function is not checked
    pattern = re.compile(f'*_{patient}.nii.gz')

    for patient in patients:
        logging.info(f"Processing patient {patient}")
        all_seg = []
        for fold in folds:
            deep_ensemble_folder = os.path.join(deep_ensemble_main, fold)
            for filename in os.listdir(deep_ensemble_folder):
                match = pattern.match(filename)
                if match:
                    logging.info(f"Loading file: {filename}")
                    seg = nib.load(os.path.join(deep_ensemble_folder, filename)).get_fdata()
                    all_seg.append(seg)           
        
        all_seg = all_seg
        shape = np.shape(all_seg)[1:] #skip first channels, this is the number of folds
        all_seg = np.array(all_seg)
        avg_pred_seg = np.zeros(shape)
        logging.info(f'Shape of all TTA outputs is {shape}')
        
        #determine most occuring ROI in each voxel
        for i in tqdm(range(shape[0])):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    vals = all_seg[:,i,j,k]
                    avg_pred_seg[i,j,k] = np.bincount(vals.astype(int)).argmax()
        
        # Save the combined segmentation as NIfTI
        combined_nii = nib.Nifti1Image(avg_pred_seg, np.eye(4))
        output_file = os.path.join(output_dir, f"combined_segmentation_patient_{patient}_TTA.nii.gz")
        nib.save(combined_nii, output_file)
        logging.info(f"Saved combined segmentation for patient {patient} at {output_file}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Combine multiple segmentation predictions into a final segmentation."
    )
    parser.add_argument("--method", type=str, default="mc_dropout",
                        help="Method of uncertainty generation (mc_dropout, ensemble, tta)")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder containing prediction files")
    parser.add_argument("--patients", type=str, nargs="+", required=True,
                        help="List of patient IDs to process")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Folder to save the combined segmentations")

    args = parser.parse_args()

    # Set default output folder
    if args.output_dir is None:
        args.output_dir = os.path.join(args.folder, "combined_segmentations")
    else:
        args.output_dir = os.path.join(args.output_dir, "combined_segmentations")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    setup_logging(args.output_dir)

    # Run the requested method
    if args.method.lower() == "mc_dropout":
        combine_mc_segmentation(args.folder, args.patients, args.output_dir)
    elif args.method.lower() == 'deep_ensemble':
        combine_de_segmentation(args.folder, args.patients, args.output_dir)
    elif args.method.lower() == 'tta':
        combine_tta_segmentation(args.folder, args.patients, args.output_dir)
    else:
        raise NotImplementedError(f"Method '{args.method}' not implemented yet")


if __name__ == "__main__":
    main()
