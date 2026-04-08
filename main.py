import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI()

# --- CORS CONFIGURATION (Allows mobile apps to connect) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BACKEND LOGIC ---

def get_farmer_advice(weight_kg, is_adult, breed_type):
    advice = []
    daily_feed = weight_kg * 0.03
    advice.append(f"<b>Feeding:</b> Needs ~{round(daily_feed, 1)} KG of dry feed daily.")
    
    if is_adult == "adult":
        target = 550 if breed_type == "beef" else 600
        if weight_kg >= target:
            advice.append("<b>Market:</b> Optimal weight reached! Ready for sale.")
        else:
            advice.append(f"<b>Market:</b> Feed for {int(target - weight_kg)} KG more for target.")
    else:
        if weight_kg > 300:
            advice.append("<b>Breeding:</b> Weight looks good for first-time breeding.")
    
    dewormer_dose = weight_kg / 50
    advice.append(f"<b>Health:</b> Dewormer dose: {round(dewormer_dose, 1)}ml (Verify with vet).")
    return advice

def process_weight_debug(image_bytes, breed_type, is_adult, strict_mode):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Invalid Image"}

    # --- MEMORY PROTECTION (Resizing for Render Free Tier) ---
    max_dim = 800
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # 1. SEGMENTATION
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. BOUNDING BOX
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "No cow detected"}
    cow_cnt = max(contours, key=cv2.contourArea)
    x, y, w_px, h_px = cv2.boundingRect(cow_cnt)

    # 3. SCALING
    standard_height_cm = 140.0 if is_adult == "adult" else 110.0
    cm_per_pixel = standard_height_cm / h_px

    # 4. MEASUREMENTS
    L_cm = round((w_px * 0.88) * cm_per_pixel, 2)
    girth_x_pos = x + int(w_px * 0.38)
    column = thresh[y : y + h_px, girth_x_pos]
    raw_depth_pixels = np.sum(column == 255) 

    max_ratio = 0.52 if strict_mode else 0.58
    refined_depth_px = min(raw_depth_pixels, h_px * max_ratio)
    D_cm = refined_depth_px * cm_per_pixel
    W_cm = D_cm * 0.85
    
    a, b = D_cm / 2, W_cm / 2
    G_cm = round(np.pi * (3*(a+b) - np.sqrt((3*a + b) * (a + 3*b))), 2)

    # 5. FORMULA
    weight_kg = (((G_cm/2.54)**2 * (L_cm/2.54)) / 300) * 0.453592
    
    # 6. SANITY FILTER
    if is_adult == "young" and weight_kg > 400:
        weight_kg = 350 + (weight_kg * 0.05)
    elif weight_kg > 900:
        weight_kg = 800 + (weight_kg * 0.05)

    return {
        "weight": round(weight_kg, 2),
        "length_cm": L_cm,
        "girth_cm": G_cm,
        "advice": get_farmer_advice(weight_kg, is_adult, breed_type)
    }

# --- ROUTES ---

@app.post("/calculate")
async def calculate_weight(file: UploadFile = File(...), breed: str = Form(...), age: str = Form(...), strict: bool = Form(False)):
    contents = await file.read()
    return process_weight_debug(contents, breed, age, strict)

@app.get("/", response_class=HTMLResponse)
async def index():
    return "<h1>SmartCow API is Online</h1><p>Use /calculate for POST requests.</p>"

if __name__ == "__main__":
    import uvicorn
    import os
    # Use the PORT provided by Render, default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)