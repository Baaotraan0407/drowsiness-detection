import glob
import os
import random
import sys
 
import cv2
import numpy as np
import tensorflow as tf
 
MODEL = os.path.join("models", "mouth_detector.h5")
DATA = os.path.join("datasets", "mouth")
 
model = tf.keras.models.load_model(MODEL)
 
 
def prob_yawn(img_bgr, rgb):
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if rgb else img_bgr
    x = cv2.resize(x, (64, 64)).astype(np.float32) / 255.0
    return float(model(np.expand_dims(x, 0), training=False).numpy()[0][0])
 
 
random.seed(0)
for label in ["no yawn", "yawn"]:
    files = glob.glob(os.path.join(DATA, label, "*"))
    sample = random.sample(files, min(50, len(files)))
    rgb, bgr, sizes = [], [], []
    for f in sample:
        img = cv2.imread(f)
        if img is None:
            continue
        sizes.append(img.shape[:2])
        rgb.append(prob_yawn(img, rgb=True))
        bgr.append(prob_yawn(img, rgb=False))
    print(f"[{label}] n={len(rgb)}  mean P(yawn): RGB input = {np.mean(rgb):.3f} | "
          f"BGR input = {np.mean(bgr):.3f}")
    print(f"    example image sizes (h, w): {sizes[:5]}")
 
if len(sys.argv) > 1:
    img = cv2.imread(sys.argv[1])
    print(f"\n{sys.argv[1]}  size={img.shape[:2]}  "
          f"P(yawn) RGB={prob_yawn(img, True):.3f}  BGR={prob_yawn(img, False):.3f}")
 