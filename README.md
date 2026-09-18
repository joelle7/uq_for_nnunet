# uq_for_nnunet

Uncertainty quantification (UQ) for a nnU-Net segmentation model. Supports MC Dropout, Deep Ensemble, and TTA-based uncertainty estimation, with tools to combine samples into a final segmentation, compute voxel-wise uncertainty maps, and evaluate uncertainty maps/calibration.

## Pipeline overview

```
 Compute N raw prediction samples per patient
 (MC Dropout / Deep Ensemble / TTA)
              │
              V
 combine_segmentation.py     -->  obtains final segmentation mask per patient (majority vote)
              │
              V
 compute_uncertainty_map.py  -->  obtains voxel-wise uncertainty maps (entropy, MI, variance, ...)
              │
              V
 evaluate_calibration.py   -->  evaluates uncertainty value calibration: ECE / accuracy vs uncertainty
 evaluate_uncertainty.py   -->  summarises uncertainty statistics per patient
```

## Installation

```bash
git clone https://github.com/joelle7/uq_for_nnunet.git
cd uq_for_nnunet/uq_for_nnunet
pip install -e .
```

This installs four command-line tools: `combine_segmentation`, `compute_uncertainty_map`, `evaluate_calibration`, and `evaluate_uncertainty`. You can also call the underlying scripts directly with `python <script>.py`.

## 0. Generating the prediction samples

This repository does **not** generate the raw prediction samples itself. It expects them to already exist in the paths.


## Expected folder structure

```
project/
├── predictions/                      # raw samples (input to the pipeline)
│   ├── case_001_01_.nii.gz            # MC Dropout: one file per sample per patient
│   ├── case_001_02_.nii.gz
│   ├── ...
│   ├── case_001_20_.nii.gz
│   └── deep_ensemble
        └── fold_0/, fold_1/, ...          # Deep Ensemble one subfolder per fold instead
│           └── case_001.npz
│   └── tta
        └── fold_0/, fold_1/, ...          # TTA: one subfolder per fold instead
│           └── case_001.npz
│
├── combined_segmentations/           # output of combine_segmentation.py
│   └── combined_segmentation_patient_001.nii.gz
│
├── uncertainty_maps/                 # output of compute_uncertainty_map.py
│   └── uncertainty_map_entropy_001.nii.gz
│
├── ground_truth/                     # optional, only needed for evaluate_calibration.py
│   └── case_001_gt.nii.gz
│
└── roi_dict.py                       # optional: {class_index: "ROI name"} dict, see below
```

The scripts locate files with a regex built from `--patients`, so check the note in **Known limitations** below if your samples aren't being picked up.


## 1. Combine segmentation samples

Combines the `N` samples per patient into one final segmentation using majority voting.

```bash
python combine_segmentation.py --method mc_dropout \
                                --folder /path/to/predictions \
                                --patients 001 002 003 \
                                #--output_dir /path/to/save/combined
```

- `--method` - mc_dropout, ensemble or tta
- `--folder` - folder containing the raw sample `.nii.gz` files
- `--patients` - one or more patient IDs to process
- `--output_dir` - where to save results (a `combined_segmentations/` subfolder is created in the /path/to/predictions if left empty)

Output: `combined_segmentations/combined_segmentation_patient_<ID>.nii.gz`

## 2. Compute uncertainty maps

Computes voxel-wise uncertainty maps (entropy, mutual information, variance, and classwise variants) from the raw `.npz` probability samples.

```bash
python compute_uncertainty_map.py --folder /path/to/npz_files \
                                   --patients 001 002 003 \
                                   --output_dir /path/to/save/uncertainty_maps \
                                   --classes 20 \
                                   --methods mc_dropout \
                                   --metrics entropy
```

Full example, with an ROI dictionary and multiple settings, written as a reusable shell script:

```bash
FOLDER="PATH_TO_PREDICTIONS_FOLDER"   # folder containing the raw samples

# Patient(s) to process. Leave empty to process all patients in the folder.
PATIENTS="001 002 003"

CLASSES=20   # number of classes in the predicted segmentation (individual structures + background)

# Optionally restrict analysis to a subset of classes (background is always kept automatically)
#KEEPING_CLASSES="3 4 5 6 7 9 10 11 12 13 14 16 17 18"

METHODS=mc_dropout          # mc_dropout, deep_ensemble, or tta
METRICS=mutual_information  # entropy, classwise_entropy, mutual_information, classwise_mutual_information, variance, classwise_variance

ROI_DICT="[...]/roi_dict.py"   # path to a Python file containing an ROI_DICT, just used for naming the saved files

OUTPUT_DIR="PATH_TO_OUTPUT"   # leave empty to save alongside the predictions

compute_uncertainty_map \
    --folder "$FOLDER" \
    --patients $PATIENTS \
    --classes "$CLASSES" \
    --metrics "$METRICS" \
    --methods "$METHODS" \
    --roi_dict "$ROI_DICT" \
    --output_dir "$OUTPUT_DIR"
    #--keep_classes $KEEPING_CLASSES
```

Output: `uncertainty_maps/uncertainty_map_<metric>_<patient>.nii.gz` - one file per class if an ROI dict is supplied (named with the ROI label), otherwise one combined multi-class file.

### ROI dictionary format

`roi_dict.py` should define a plain dictionary mapping class index to a readable ROI name, e.g.:

```python
ROI_DICT = {
    0: "Background",
    1: "Carotid artery (L)",
    2: "Carotid artery (R)",
    # ...
}
```

## 3. Evaluate results

### Evaluate calibration

Computes reliability/calibration statistics (Expected Calibration Error) by comparing uncertainty against prediction accuracy relative to ground truth.

```bash
UQ_folder="[...]/uncertainty_maps"
PRED_folder="path/to/predictions"
GT_folder="path/to/ground_truth"
OUTPUT_EXCEL_CALIBRATION="EXCEL_OUTPUT_PATH.xlsx"

PATIENTS="001 002 003"  # optional - leave empty to process all patients in the folder

METRIC = entropy
evaluate_calibration \
    --UQ_folder $UQ_folder \
    --Pred_folder $PRED_folder \
    --GT_folder $GT_folder \
    --output_excel $OUTPUT_EXCEL_CALIBRATION
    --METRIC
    #--patients $PATIENTS
```

### Evaluate uncertainty (generate uncertainty scores)

Computes summary statistics (mean, std, percentiles, proportion of voxels above various thresholds) for each patient's uncertainty map and writes them to Excel.

```bash
METRIC = entropy

evaluate_uncertainty \
    --UQ_folder $UQ_folder \
    --output_excel $OUTPUT_EXCEL_UNCERTAINTY_SCORES
    --metric $METRIC
    #--patients $PATIENTS
```

## Known limitations

This is an first release of the research code. For any questions or comments please contact :) (j.e.van.aalst[at]umcg.nl/joelle.vanaalst[at]live.nl)
- **Folder structure samples** the folder structure for the mc_dropout samples are directly in the predictions folder and for Test-time augmentation and Deep ensemble must have subfolders. TO DO: fix
- **Filename mismatch between steps 2 and 3.** `evaluate_calibration.py`/`evaluate_uncertainty.py` currently look for files by formatting, this needs to be fixed to a more robust method.
- **Pass `--roi_dict` for now** when running `compute_uncertainty_map.py`, it needs to be present and cannot be left blank for now
- **MC Dropout sample matching is capped at 20 samples**

