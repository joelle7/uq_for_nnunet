#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Description:    Convert nnU-Net raw prediction samples from .npz (nnU-Net's
                 default output when save_probabilities=True) to .nii.gz, so
                 they can be used by combine_segmentation.py and
                 compute_uncertainty_map.py.

                 Each .npz file is expected to contain a 'probabilities' array
                 of shape (n_classes, X, Y, Z), which is nnU-Net's default key.

                 Two things can be written per .npz file:
                   - an argmax (hard) label map, one .nii.gz file, named the
                     same as the .npz file. This is what combine_segmentation.py
                     expects as input for its majority vote.
                   - one probability map per ROI, named using roi_dict, in the
                     same style as compute_uncertainty_map.py's uncertainty maps.

Usage:
    python convert_npz_to_nifti.py --folder /path/to/npz_files \
                                    --output_dir /path/to/save/nifti \
                                    --roi_dict /path/to/roi_dict.py \
                                    --save_argmax --save_per_roi

Arguments:
    --folder        Folder containing .npz prediction files
    --output_dir    Folder to save the converted .nii.gz file(s)
    --roi_dict      Path to a Python file containing ROI_DICT (only needed
                     for --save_per_roi)
    --save_argmax   Save the argmax hard segmentation (default: on)
    --save_per_roi  Additionally save one probability map per ROI
    --key           Key inside the .npz file to read (default: 'probabilities')

===============================================================================
"""

import os
import glob
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import nibabel as nib
from tqdm import tqdm

from uq_for_nnunet.utils.Transforms.transformer import Transformer


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
    log_file = os.path.join(output_dir, f"log_convert_npz_to_nifti_{timestamp}.txt")

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


def load_roi_dict(roi_dict_path: str) -> dict:
    """
    Load a ROI_DICT from a Python file, e.g. utils/roi_dict.py.

    Args:
        roi_dict_path (str): path to a .py file defining ROI_DICT = {class_index: "name"}.

    Returns:
        dict: the loaded ROI_DICT.
    """
    import runpy
    module_vars = runpy.run_path(roi_dict_path)
    if "ROI_DICT" not in module_vars:
        raise ImportError(f"Provided roi_dict file {roi_dict_path} does not define ROI_DICT")
    return module_vars["ROI_DICT"]


def convert_npz_to_nifti(
    npz_file: str,
    output_dir: str,
    roi_dict: dict = None,
    affine: np.ndarray = np.eye(4),
    save_argmax: bool = True,
    save_per_roi: bool = False,
    key: str = "probabilities",
) -> list:
    """
    Convert a single nnU-Net .npz prediction sample into .nii.gz file(s).

    Args:
        npz_file (str): path to the .npz file to convert.
        output_dir (str): folder the .nii.gz file(s) will be written into.
        roi_dict (dict, optional): {class_index: "ROI name"}, see utils/roi_dict.py.
            Required when save_per_roi=True; only used for naming the output files.
        affine (np.ndarray): affine matrix written into the NIfTI header.
            Defaults to identity, matching the rest of this repo (see
            compute_uncertainty_map.py) since nnU-Net's own .pkl sidecar with
            the true spacing/origin/direction is not read here.
        save_argmax (bool): if True, save the argmax (hard) label map as a
            single .nii.gz file, named identically to the .npz file. This is
            the format combine_segmentation.py expects for its raw samples.
        save_per_roi (bool): if True, additionally save one probability map
            per ROI, named "<npz_basename>_<roi_name>.nii.gz", in the same
            style as compute_uncertainty_map.py's save_uncertainty_map_as_nifti.
        key (str): key of the array to read from the .npz file. nnU-Net's
            default is 'probabilities' (shape: n_classes, X, Y, Z).

    Returns:
        list[str]: paths of all .nii.gz files written.
    """
    if save_per_roi and not roi_dict:
        raise ValueError("roi_dict must be provided when save_per_roi=True")

    os.makedirs(output_dir, exist_ok=True)
    transformer = Transformer()

    data = np.load(npz_file)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_file}. Available keys: {list(data.keys())}")

    probabilities = data[key]  # shape: (n_classes, X, Y, Z)
    basename = Path(npz_file).stem  # drops the .npz extension
    written_files = []

    if save_argmax:
        segmentation = np.argmax(probabilities, axis=0).astype(np.uint8)  # (X, Y, Z)
        segmentation = transformer.transform_uncertainty_map(segmentation)
        out_file = os.path.join(output_dir, f"{basename}.nii.gz")
        nib.save(nib.Nifti1Image(segmentation, affine), out_file)
        logging.info(f"Saved argmax segmentation to {out_file}")
        written_files.append(out_file)

    if save_per_roi:
        for c in range(probabilities.shape[0]):
            roi_name = roi_dict.get(c, f"class{c}")
            class_map = probabilities[c]  # (X, Y, Z)
            class_map = transformer.transform_uncertainty_map(class_map)
            out_file = os.path.join(output_dir, f"{basename}_{roi_name}.nii.gz")
            nib.save(nib.Nifti1Image(class_map, affine), out_file)
            logging.info(f"Saved '{roi_name}' probability map to {out_file}")
            written_files.append(out_file)

    return written_files


def convert_folder(
    folder: str,
    output_dir: str,
    roi_dict: dict = None,
    save_argmax: bool = True,
    save_per_roi: bool = False,
    key: str = "probabilities",
    recursive: bool = False,
) -> list:
    """
    Convert every .npz file in a folder to .nii.gz. Set recursive=True to also
    pick up .npz files in subfolders (e.g. the fold_0/, fold_1/, ... layout
    used for deep_ensemble / tta).

    Args:
        folder (str): folder containing .npz files.
        output_dir (str): folder to write the .nii.gz file(s) into.
        roi_dict (dict, optional): see convert_npz_to_nifti().
        save_argmax (bool): see convert_npz_to_nifti().
        save_per_roi (bool): see convert_npz_to_nifti().
        key (str): see convert_npz_to_nifti().
        recursive (bool): search subfolders for .npz files too.

    Returns:
        list[str]: paths of all .nii.gz files written.
    """
    pattern = "**/*.npz" if recursive else "*.npz"
    npz_files = sorted(glob.glob(os.path.join(folder, pattern), recursive=recursive))

    if not npz_files:
        logging.warning(f"No .npz files found in {folder} (recursive={recursive})")
        return []

    written_files = []
    for npz_file in tqdm(npz_files, desc="Converting samples"):
        written_files.extend(
            convert_npz_to_nifti(
                npz_file=npz_file,
                output_dir=output_dir,
                roi_dict=roi_dict,
                save_argmax=save_argmax,
                save_per_roi=save_per_roi,
                key=key,
            )
        )
    return written_files


def main():
    parser = argparse.ArgumentParser(description="Convert nnU-Net .npz prediction samples to .nii.gz.")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .npz prediction files")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save the converted .nii.gz files")
    parser.add_argument("--roi_dict", type=str, default=None, help="Path to Python file containing ROI_DICT (e.g., roi_dict.py)")
    parser.add_argument("--save_argmax", action="store_true", default=True, help="Save the argmax hard segmentation (default: on)")
    parser.add_argument("--no_save_argmax", dest="save_argmax", action="store_false", help="Skip saving the argmax hard segmentation")
    parser.add_argument("--save_per_roi", action="store_true", default=False, help="Additionally save one probability map per ROI (requires --roi_dict)")
    parser.add_argument("--key", type=str, default="probabilities", help="Key inside the .npz file to read (default: 'probabilities')")
    parser.add_argument("--recursive", action="store_true", default=False, help="Also search subfolders (e.g. fold_0/, fold_1/, ...) for .npz files")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.folder, "nifti_samples")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    roi_dict_data = None
    if args.roi_dict:
        roi_dict_data = load_roi_dict(args.roi_dict)
        logging.info(f"Loaded ROI_DICT: {roi_dict_data}")
    elif args.save_per_roi:
        raise ValueError("--roi_dict is required when --save_per_roi is set")

    written_files = convert_folder(
        folder=args.folder,
        output_dir=args.output_dir,
        roi_dict=roi_dict_data,
        save_argmax=args.save_argmax,
        save_per_roi=args.save_per_roi,
        key=args.key,
        recursive=args.recursive,
    )
    logging.info(f"Done. Wrote {len(written_files)} .nii.gz file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
