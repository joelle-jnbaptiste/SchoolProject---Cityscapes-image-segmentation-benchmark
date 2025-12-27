import io
import base64
import torch
import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", DEVICE)

app = FastAPI(title="Dual Segmentation API")

# -----------------------------
# PREPROCESSING
# -----------------------------
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def load_image_bytes(raw_bytes):
    return np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"), dtype=np.uint8)

def resize_by_height(img, target_height=256):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    img_resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
    return img_resized


def preprocess_deeplab(img_np):
    # 1. Convert RGB numpy
    img = img_np.astype(np.float32)

    # 2. Resize par hauteur = 256, ratio conservé
    h, w = img.shape[:2]
    scale = 256 / h
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, 256), interpolation=cv2.INTER_LINEAR)

    # 3. Normalisation
    img = img / 255.0
    img = (img - MEAN) / STD

    # 4. CHW + batch
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)

    return img



def preprocess_mask2former(img_np, processor):
    return processor(images=img_np, return_tensors="pt")["pixel_values"].to(DEVICE)

# -----------------------------
# POSTPROCESSING
# -----------------------------
palette = np.array([
    [128,  64,128],
    [244,  35,232],
    [ 70,  70, 70],
    [102, 102,156],
    [190, 153,153],
    [153, 153,153],
    [250, 170, 30],
    [220, 220,  0],
], dtype=np.uint8)

IGNORE = 255

def colorize(mask):
    H, W = mask.shape
    img = np.zeros((H,W,3), dtype=np.uint8)
    valid = mask != IGNORE
    img[valid] = palette[mask[valid]]
    return img

def to_base64(img_np):
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# -----------------------------
# REMAP
# -----------------------------
remap_dict = {i:i for i in range(8)}

def remap(mask_np):
    out = np.full_like(mask_np, IGNORE)
    for k,v in remap_dict.items():
        out[mask_np == k] = v
    return out

# -----------------------------
# MODEL BUILDERS
# -----------------------------
NUM_CLASSES = 8

def make_model_deeplab(aux_loss=True):
    model = deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        aux_loss=aux_loss
    )
    model.classifier[4] = torch.nn.Conv2d(
        model.classifier[4].in_channels, NUM_CLASSES, 1
    )
    if aux_loss:
        model.aux_classifier[4] = torch.nn.Conv2d(
            model.aux_classifier[4].in_channels, NUM_CLASSES, 1
        )
    return model

def make_model_mask2former():
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-small-coco-panoptic",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

# -----------------------------
# LOAD MODELS
# -----------------------------
# DeepLab 
deeplab = make_model_deeplab().to(DEVICE)
deeplab.load_state_dict(torch.load("model/deeplab.pt", map_location=DEVICE))
deeplab.eval()

# Mask2Former
processor = Mask2FormerImageProcessor.from_pretrained(
    "facebook/mask2former-swin-small-coco-panoptic"
)

mask2former = make_model_mask2former().to(DEVICE)
mask2former.load_state_dict(torch.load("model/mask2former.pt", map_location=DEVICE))
mask2former.eval()

# -----------------------------
# PREDICT FUNCTIONS
# -----------------------------
def predict_deeplab(img_np):
    tens = preprocess_deeplab(img_np).to(DEVICE)
    with torch.no_grad():
        out = deeplab(tens)["out"]
        pred = torch.argmax(out, dim=1)[0].cpu().numpy()
    return remap(pred)

def predict_mask2former(img_np):
    tens = preprocess_mask2former(img_np, processor)
    with torch.no_grad():
        out = mask2former(pixel_values=tens)
        seg = processor.post_process_semantic_segmentation(
            out,
            target_sizes=[img_np.shape[:2]]
        )[0]
        return remap(seg.cpu().numpy())

# -----------------------------
# ROUTE
# -----------------------------
@app.post("/predict")
async def predict_route(image: UploadFile):
    raw = await image.read()
    img_np = load_image_bytes(raw)
    
    m1 = predict_deeplab(img_np)
    m2 = predict_mask2former(img_np)

    return JSONResponse({
        "deeplab_png": to_base64(colorize(m1)),
        "mask2former_png": to_base64(colorize(m2))
    })
