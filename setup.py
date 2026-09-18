from setuptools import setup, find_packages

setup(
    name="uq_for_nnunet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "nibabel",
        "tqdm",
        "pandas",
        "openpyxl",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "combine_segmentation = uq_for_nnunet.sample_combination.combine_segmentation:main",
            "compute_uncertainty_map = uq_for_nnunet.sample_combination.compute_uncertainty_map:main",
            "evaluate_calibration = uq_for_nnunet.evaluation.evaluate_calibration:main",
            "evaluate_uncertainty = uq_for_nnunet.evaluation.evaluate_uncertainty:main"
        ],
    },
)
