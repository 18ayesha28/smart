"""
Smart NeuroCare — Dataset download & preparation.

Downloads both datasets via the Kaggle API and reorganizes them into the
folder layouts the training scripts expect:

  data/classification/
      train/{glioma,meningioma,pituitary,notumor}/*.jpg
      val/{glioma,meningioma,pituitary,notumor}/*.jpg
      test/{glioma,meningioma,pituitary,notumor}/*.jpg   (kept untouched from source "Testing/")

  data/detection/            (binary — merges the 3 tumor classes)
      train/{tumor,no_tumor}/*.jpg
      val/{tumor,no_tumor}/*.jpg

  data/segmentation/
      <patient_id>/*.tif + *_mask.tif   (kept as-is, split handled at load time by patient)

Run:
    pip install kaggle
    python prepare_data.py
"""

import os
import shutil
import random
import subprocess

random.seed(42)

RAW_DIR = "data_raw"
OUT_DIR = "data"
VAL_FRACTION = 0.15


def download_datasets():
    os.makedirs(f"{RAW_DIR}/classification", exist_ok=True)
    os.makedirs(f"{RAW_DIR}/segmentation", exist_ok=True)

    print("Downloading classification dataset (masoudnickparvar/brain-tumor-mri-dataset)...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "masoudnickparvar/brain-tumor-mri-dataset",
        "-p", f"{RAW_DIR}/classification", "--unzip",
    ], check=True)

    print("Downloading segmentation dataset (mateuszbuda/lgg-mri-segmentation)...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "mateuszbuda/lgg-mri-segmentation",
        "-p", f"{RAW_DIR}/segmentation", "--unzip",
    ], check=True)


def find_classification_root():
    """The zip layout varies slightly by upload; locate the folder containing
    Training/ and Testing/ subfolders."""
    for root, dirs, _ in os.walk(f"{RAW_DIR}/classification"):
        if "Training" in dirs and "Testing" in dirs:
            return root
    raise FileNotFoundError("Could not locate Training/Testing folders in downloaded dataset")


def prepare_classification_and_detection():
    src_root = find_classification_root()
    classes = ["glioma", "meningioma", "pituitary", "notumor"]

    # Kaggle class folder names sometimes have suffixes like "glioma_tumor" — normalize
    def resolve_class_dir(base, cls):
        candidates = [d for d in os.listdir(base) if cls in d.lower()]
        if not candidates:
            raise FileNotFoundError(f"No folder matching '{cls}' under {base}")
        return os.path.join(base, candidates[0])

    train_src = os.path.join(src_root, "Training")
    test_src = os.path.join(src_root, "Testing")

    print("\nClass counts (Training):")
    for cls in classes:
        cls_dir = resolve_class_dir(train_src, cls)
        files = os.listdir(cls_dir)
        print(f"  {cls}: {len(files)} images")

        # Split into train/val
        random.shuffle(files)
        n_val = int(len(files) * VAL_FRACTION)
        val_files, train_files = files[:n_val], files[n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files)]:
            # 4-class layout (for classification model)
            dest = os.path.join(OUT_DIR, "classification", split_name, cls)
            os.makedirs(dest, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(cls_dir, f), os.path.join(dest, f))

            # binary layout (for detection model): notumor -> no_tumor, others -> tumor
            binary_label = "no_tumor" if cls == "notumor" else "tumor"
            bin_dest = os.path.join(OUT_DIR, "detection", split_name, binary_label)
            os.makedirs(bin_dest, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(cls_dir, f), os.path.join(bin_dest, f))

    # Keep original Testing/ folder as held-out test set (classification)
    print("\nClass counts (Testing / held-out):")
    for cls in classes:
        cls_dir = resolve_class_dir(test_src, cls)
        files = os.listdir(cls_dir)
        print(f"  {cls}: {len(files)} images")

        dest = os.path.join(OUT_DIR, "classification", "test", cls)
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(cls_dir, f), os.path.join(dest, f))

        binary_label = "no_tumor" if cls == "notumor" else "tumor"
        bin_dest = os.path.join(OUT_DIR, "detection", "test", binary_label)
        os.makedirs(bin_dest, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(cls_dir, f), os.path.join(bin_dest, f))


def prepare_segmentation():
    src_candidates = [
        d for d in os.listdir(f"{RAW_DIR}/segmentation")
        if os.path.isdir(os.path.join(f"{RAW_DIR}/segmentation", d))
    ]
    # lgg-mri-segmentation typically unzips to a folder like "kaggle_3m" containing
    # one subfolder per patient
    src_root = None
    for candidate in src_candidates:
        candidate_path = os.path.join(f"{RAW_DIR}/segmentation", candidate)
        subdirs = [d for d in os.listdir(candidate_path) if os.path.isdir(os.path.join(candidate_path, d))]
        if any(d.startswith("TCGA") for d in subdirs):
            src_root = candidate_path
            break

    if src_root is None:
        raise FileNotFoundError("Could not locate patient folders (TCGA_*) in segmentation dataset")

    dest_root = os.path.join(OUT_DIR, "segmentation")
    os.makedirs(dest_root, exist_ok=True)

    patient_dirs = [d for d in os.listdir(src_root) if d.startswith("TCGA")]
    print(f"\nFound {len(patient_dirs)} patients in segmentation dataset. Copying...")

    for patient in patient_dirs:
        src_patient_dir = os.path.join(src_root, patient)
        dest_patient_dir = os.path.join(dest_root, patient)
        if os.path.exists(dest_patient_dir):
            continue
        shutil.copytree(src_patient_dir, dest_patient_dir)

    print(f"Segmentation data ready at: {dest_root}")


if __name__ == "__main__":
    download_datasets()
    prepare_classification_and_detection()
    prepare_segmentation()
    print("\nDone. Folder layout:")
    print("  data/classification/{train,val,test}/{glioma,meningioma,pituitary,notumor}/")
    print("  data/detection/{train,val,test}/{tumor,no_tumor}/")
    print("  data/segmentation/<patient_id>/*.tif")
