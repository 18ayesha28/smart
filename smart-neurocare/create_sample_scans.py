"""
Generate realistic synthetic brain MRI test slices for instant 1-click clinical testing.
"""
import os
import cv2
import numpy as np
from PIL import Image

def generate_sample_mris(output_dir="sample_scans"):
    os.makedirs(output_dir, exist_ok=True)

    samples = {
        "meningioma_sample.png": {"tumor": True, "type": "meningioma", "loc": (175, 110), "r": 28, "bright": 210},
        "glioma_sample.png": {"tumor": True, "type": "glioma", "loc": (95, 145), "r": 34, "bright": 195},
        "pituitary_sample.png": {"tumor": True, "type": "pituitary", "loc": (128, 140), "r": 22, "bright": 225},
        "healthy_normal_sample.png": {"tumor": False, "type": "normal", "loc": None, "r": 0, "bright": 0},
        # --- DEMO / SIMULATED FOLLOW-UP SCANS ---
        # These are NOT real patient follow-up data. They exist solely to let a
        # reviewer demonstrate the longitudinal treatment-response feature (two
        # "visits" for the same demo patient_id) without needing a real serial-MRI
        # dataset. Each is watermarked "SIMULATED FOLLOW-UP" directly on the image
        # and must never be presented as a real clinical scan.
        "glioma_followup_smaller_demo.png": {
            "tumor": True, "type": "glioma", "loc": (95, 145), "r": 17, "bright": 195,
            "demo_watermark": True,
        },
        "glioma_followup_larger_demo.png": {
            "tumor": True, "type": "glioma", "loc": (95, 145), "r": 46, "bright": 195,
            "demo_watermark": True,
        },
    }

    for filename, config in samples.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            continue

        # Create realistic synthetic 256x256 brain MRI slice
        img = np.zeros((256, 256), dtype=np.uint8)

        # Draw skull contour (outer bone ring)
        cv2.ellipse(img, (128, 128), (95, 115), 0, 0, 360, 180, 2)

        # Draw outer brain parenchyma (cerebral cortex)
        cv2.ellipse(img, (128, 128), (88, 108), 0, 0, 360, 90, -1)

        # Add brain tissue texture (gyri/sulci convolutions)
        noise = np.random.normal(0, 12, (256, 256)).astype(np.float32)
        brain_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(brain_mask, (128, 128), (87, 107), 0, 0, 360, 255, -1)

        img_float = img.astype(np.float32)
        img_float[brain_mask > 0] += noise[brain_mask > 0]

        # Draw ventricles (central CSF cavities - hypointense on T1)
        cv2.ellipse(img_float, (116, 120), (8, 26), -8, 0, 360, 30, -1)
        cv2.ellipse(img_float, (140, 120), (8, 26), 8, 0, 360, 30, -1)

        # Interhemispheric fissure
        cv2.line(img_float, (128, 30), (128, 226), 35, 1)

        # If tumor slice, draw lesion (hyperintense post-contrast T1)
        if config["tumor"]:
            cx, cy = config["loc"]
            r = config["r"]
            # Main hyperintense core
            cv2.circle(img_float, (cx, cy), r, config["bright"], -1)
            # Perilesional edema / blurred boundary
            blur_kernel = (r*2 + 1, r*2 + 1)
            edema_mask = np.zeros((256, 256), dtype=np.float32)
            cv2.circle(edema_mask, (cx, cy), int(r * 1.3), 50.0, -1)
            img_float += edema_mask

        # Apply realistic Gaussian smoothing
        img_smooth = cv2.GaussianBlur(img_float, (5, 5), 1.2)
        img_final = np.clip(img_smooth, 0, 255).astype(np.uint8)

        # Burn an unmistakable watermark into simulated follow-up scans so the
        # image itself can never be mistaken for real patient data, even if it
        # were exported/shared outside the app.
        img_rgb = cv2.cvtColor(img_final, cv2.COLOR_GRAY2BGR)
        if config.get("demo_watermark"):
            cv2.rectangle(img_rgb, (0, 0), (256, 22), (0, 0, 0), -1)
            cv2.putText(img_rgb, "SIMULATED FOLLOW-UP - DEMO ONLY", (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)

        # Convert to RGB and save
        Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)).save(filepath)

    return samples

if __name__ == "__main__":
    generate_sample_mris()
    print("Sample MRIs created successfully.")
