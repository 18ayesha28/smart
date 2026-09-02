# Running Smart NeuroCare Demo on Windows 11

This gets you a working local demo UI in your browser. No cloud, no GPU required
(models run on CPU — slow but fine for a handful of images).

## 1. Install Python (if you don't have it)

1. Download Python 3.11 from https://www.python.org/downloads/windows/
2. Run the installer. **Check "Add python.exe to PATH"** before clicking Install.
3. Verify it worked — open **PowerShell** (search "PowerShell" in Start menu) and run:
   ```powershell
   python --version
   ```
   You should see something like `Python 3.11.x`.

## 2. Put all the files in one folder

Create a folder, e.g. `C:\Users\<you>\Documents\smart-neurocare`, and place all
downloaded files in it:
```
smart-neurocare/
  app.py
  cnn_detection_model.py
  unet_segmentation.py
  hospital_recommendation.py
  report_generator.py
  api_scans.py
  requirements.txt
```

## 3. Open PowerShell in that folder

Easiest way: open the folder in File Explorer, click the address bar, type `powershell`,
and press Enter. It opens PowerShell already pointed at that folder.

## 4. Create a virtual environment and install dependencies

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If `venv\Scripts\activate` gives a "running scripts is disabled" error, run this once
(PowerShell as normal user, not admin needed):
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
then re-run `venv\Scripts\activate`.

You'll know it worked when your prompt shows `(venv)` at the start of the line.

## 5. Run the demo

```powershell
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) and should
open it automatically in your browser. If not, copy the URL into your browser manually.

## 6. Using it

- Upload any brain MRI image (PNG/JPEG — you can find sample images by searching
  "brain MRI png" or by using images from the Kaggle "Brain MRI Images for Brain
  Tumor Detection" dataset).
- Click **Run Analysis**.
- You'll see a (demo, untrained) detection result, a segmentation overlay,
  estimated metrics, a downloadable PDF report, and hospital recommendations.

## Stopping the app

Go back to the PowerShell window and press `Ctrl+C`.

## Next time you want to run it

You don't need to reinstall anything — just:
```powershell
cd C:\Users\<you>\Documents\smart-neurocare
venv\Scripts\activate
streamlit run app.py
```

## What's real vs. placeholder in this demo

| Piece | Status |
|---|---|
| Upload UI, pipeline wiring, PDF report, hospital scoring math | Fully functional |
| Detection/segmentation model architectures | Real (PyTorch, EfficientNet/U-Net) |
| Detection/segmentation model **weights** | Random/untrained — results are not medically meaningful |
| Hospital data | 3 hardcoded sample hospitals, not a real database |
| Auth, database, cloud storage, queues (from the architecture doc) | Not included in this demo — this is a local single-user prototype |

To make results medically meaningful, you'd train the models in
`cnn_detection_model.py` / `unet_segmentation.py` on a labeled dataset (see the
"Datasets" section of the design doc) and load the trained checkpoint in `app.py`.
