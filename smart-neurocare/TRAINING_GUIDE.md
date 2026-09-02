# Training Guide — Smart NeuroCare Models (From Scratch)

## Why your model always says "no tumor" at ~50% confidence

This isn't a bug in the architecture — it's expected behavior for an **untrained**
network. In `app.py` the models are created like this:

```python
detection_model = TumorDetectionModel(pretrained=False)
```

`pretrained=False` means the final classification layer has **random weights**.
A randomly-initialized linear layer, fed into a sigmoid, outputs a value close
to 0.5 for almost everything, regardless of input — because it hasn't learned
any pattern yet. Since the threshold is `prob > 0.5`, and random weights hover
right around that boundary, you get "no tumor" (or a coin-flip) on every image.

**The fix is not a code fix — it's training the model on labeled data.** Below
is the full path from zero to a working, evaluated checkpoint you can drop into
`app.py`.

---

## 1. Datasets to use

You need two different datasets because detection/classification and
segmentation are different tasks with different label types.

### A) Detection + Classification: **Brain Tumor MRI Dataset**
- Kaggle: `https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset`
- 7,023 MRI images, already split into `Training/` and `Testing/` folders,
  each with 4 class subfolders: `glioma`, `meningioma`, `pituitary`, `notumor`.
- This single dataset covers **both** your detection model (tumor vs. no
  tumor — just merge the 3 tumor folders) and your classification model
  (which type of tumor).
- Well-established, widely used, good size — this is the right starting
  point, not the smaller 253-image Kaggle "Brain MRI Images for Brain Tumor
  Detection" dataset (too small to generalize well).

### B) Segmentation: **LGG MRI Segmentation Dataset**
- Kaggle: `https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation`
- 110 patients' worth of brain MRI slices, each with a **manually-annotated
  tumor mask** (`.tif` image + `_mask.tif` pixel mask), sourced from TCIA/TCGA
  lower-grade glioma collection.
- This is the dataset that actually has pixel-level ground truth, which is
  required to train a U-Net. (The classification dataset above has no masks
  — you cannot train segmentation on it.)
- Folder structure: one folder per patient (e.g. `TCGA_CS_4941_19960909/`),
  each containing paired image/mask `.tif` files per slice.

> Both are free, no special access request needed, and both come from
> legitimate clinical sources (Figshare/TCGA-derived), not random web images.

---

## 2. Getting the datasets onto your Windows machine

1. Create a free Kaggle account: https://www.kaggle.com
2. Go to your Kaggle account settings → "Create New Token" → downloads
   `kaggle.json`. This is your API key.
3. In PowerShell, in your project folder:
   ```powershell
   pip install kaggle
   mkdir $env:USERPROFILE\.kaggle
   copy C:\path\to\downloaded\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
   ```
4. Download both datasets (run `prepare_data.py`, provided below — it wraps
   these calls and reorganizes the folders for you):
   ```powershell
   kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data_raw/classification --unzip
   kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data_raw/segmentation --unzip
   ```

---

## 3. Local (CPU) vs. Google Colab (free GPU)

Training on a Windows laptop CPU **will work** for the detection/classification
model (small 2D images, EfficientNet-B0) but expect **hours per run** instead
of minutes. Segmentation (U-Net over thousands of slices) is genuinely painful
on CPU.

**Recommendation:** do the actual training in **Google Colab** (free GPU,
zero setup), then download the resulting `.pt` checkpoint file and use it
locally in your Windows `app.py` for inference only (inference on CPU is fast
— it's training that's slow).

Steps:
1. Go to https://colab.research.google.com, new notebook.
2. Runtime → Change runtime type → GPU (T4).
3. Upload `cnn_detection_model.py`, `unet_segmentation.py`,
   `train_detection.py`, `train_segmentation.py`, and your `kaggle.json` to
   the Colab session (left sidebar → Files → upload).
4. Run:
   ```python
   !pip install kaggle
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !python prepare_data.py
   !python train_detection.py
   !python train_segmentation.py
   ```
5. Download `best_detection_model.pt`, `best_classification_model.pt`, and
   `best_segmentation_model.pt` from Colab's file browser to your Windows
   project folder.
6. Point `app.py` at them (see the "Wiring trained weights into app.py"
   section at the bottom).

If you'd rather train locally on Windows anyway (just slower), everything
below runs the same way — just `python train_detection.py` in your activated
`venv`.

---

## 4. What "properly train" actually means here

- **Train/validation/test split**: never evaluate on data the model trained
  on. The classification dataset already ships with Training/Testing folders
  — further split Training into train/val (e.g. 85/15) so you can tune without
  touching the test set.
- **Class balance**: check how many images per class before training
  (`prepare_data.py` prints this). If wildly imbalanced, use class weights
  (already wired into the loss function) rather than just accuracy as your
  metric.
- **Patient-level split for segmentation**: never put slices from the same
  patient in both train and validation — that leaks information and inflates
  your Dice score. `train_segmentation.py` splits by patient folder, not by
  individual slice.
- **Enough epochs + early stopping**: 15–30 epochs is a reasonable starting
  range for transfer learning on this size of dataset; the training scripts
  save the best checkpoint by validation loss automatically, so you can set
  epochs generously and just keep the best one.
- **Evaluate with more than accuracy**: precision/recall/F1/confusion matrix
  for classification, Dice/IoU for segmentation. Both training scripts print
  these.

---

## 5. Wiring trained weights into `app.py`

Once you have `best_detection_model.pt`, `best_classification_model.pt`, and
`best_segmentation_model.pt` in your project folder, update the model loading
in `app.py`:

```python
@st.cache_resource
def load_models():
    detection_model = TumorDetectionModel(pretrained=False)
    detection_model.load_state_dict(torch.load("best_detection_model.pt", map_location="cpu"))
    detection_model.eval()

    classification_model = TumorClassificationModel(num_classes=3)
    classification_model.load_state_dict(torch.load("best_classification_model.pt", map_location="cpu"))
    classification_model.eval()

    segmentation_model = UNet(in_channels=1, out_channels=1)
    segmentation_model.load_state_dict(torch.load("best_segmentation_model.pt", map_location="cpu"))
    segmentation_model.eval()

    return detection_model, classification_model, segmentation_model
```

This single change is what takes the app from "demo, meaningless outputs" to
"actually reflects what the models learned from real labeled MRI data."
It is still not a clinically validated diagnostic tool — but the confidence
scores will now mean something.
