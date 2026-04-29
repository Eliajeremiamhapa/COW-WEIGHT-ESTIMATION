import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import math
import httpx
import asyncio

app = FastAPI()

# MUHIMU KWA MOBILE APP: Inaruhusu App yako kuongea na API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SEHEMU YA KUIAMSHA SERVER (KEEP-ALIVE) ---
async def keep_server_awake():
    """Inatuma ping kila baada ya dakika 10 kuzuia Render isilale"""
    await asyncio.sleep(10) # Subiri kidogo baada ya kuwaka
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Inajitumia request yenyewe
                await client.get("https://cow-weight-estimation.onrender.com/")
                print("Server Keep-Alive: Ping successful")
            except Exception as e:
                print(f"Keep-Alive Error: {e}")
            await asyncio.sleep(600) # Subiri dakika 10

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(keep_server_awake())
# ----------------------------------------------

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def get_farmer_advice(weight_kg, is_adult, breed_type):
    advice = []
    daily_feed = weight_kg * 0.03
    advice.append(f"<b>Feeding:</b> Needs ~{round(daily_feed, 1)} KG dry feed daily.")
    
    if is_adult == "adult":
        target = 550 if breed_type == "beef" else 600
        if weight_kg >= target:
            advice.append("<b>Market:</b> Optimal weight reached! Ready for sale.")
        else:
            advice.append(f"<b>Market:</b> Feed for {int(target - weight_kg)} KG more.")
    else:
        if weight_kg > 300:
            advice.append("<b>Breeding:</b> Ready for first-time breeding.")
    
    dewormer_dose = weight_kg / 50
    advice.append(f"<b>Health:</b> Dewormer dose: {round(dewormer_dose, 1)}ml.")
    return advice

def process_weight_final_v4(image_bytes, breed_type, is_adult, strict_mode):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return {"error": "Invalid Image"}

    # Resize kwa ajili ya seva ndogo (Render Free Tier)
    img = cv2.resize(img, (800, 600))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return {"error": "No cow detected"}
    cow_cnt = max(contours, key=cv2.contourArea)
    x, y, w_px, h_px = cv2.boundingRect(cow_cnt)
    
    x = int(x)
    y = int(y)
    w_px = int(w_px)
    h_px = int(h_px)

    # ============ CALIBRATION ============
    BASE_CALIBRATION = 0.54
    CALIBRATION_FACTOR = BASE_CALIBRATION
    
    # ============ LENGTH (L) CALCULATION ============
    body_length_pixels = w_px * 0.65
    L_cm = body_length_pixels * CALIBRATION_FACTOR
    L_cm = float(round(L_cm, 2))
    
    # ============ GIRTH (G) CALCULATION ============
    girth_x = x + int(w_px * 0.38)
    
    if girth_x >= thresh.shape[1]:
        girth_x = thresh.shape[1] - 1
    
    column = thresh[y:y+h_px, girth_x]
    
    white_indices = np.where(column == 255)[0]
    
    if len(white_indices) > 0:
        bottom_start = int(h_px * 0.7)
        bottom_part = white_indices[white_indices > bottom_start]
        
        if len(bottom_part) > 0:
            max_segment = 1
            current = 1
            for i in range(1, len(bottom_part)):
                if bottom_part[i] == bottom_part[i-1] + 1:
                    current += 1
                else:
                    max_segment = max(max_segment, current)
                    current = 1
            max_segment = max(max_segment, current)
            depth_pixels = max_segment
        else:
            depth_pixels = 30
    else:
        depth_pixels = 30
    
    max_allowed_depth = int(h_px * 0.20)
    depth_pixels = min(depth_pixels, max_allowed_depth)
    depth_pixels = max(depth_pixels, 25)
    
    depth_cm = depth_pixels * CALIBRATION_FACTOR
    width_cm = depth_cm * 0.60
    
    a = depth_cm / 2
    b = width_cm / 2
    h_el = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0
    G_cm = math.pi * (a + b) * (1 + (3 * h_el) / (10 + math.sqrt(4 - 3 * h_el)))
    G_cm = float(round(G_cm, 2))
    
    # ============ WEIGHT CALCULATION ============
    raw_weight = (G_cm ** 2 * L_cm) / 10000
    FINAL_ADJUSTMENT = 1.117
    
    weight_kg = raw_weight * FINAL_ADJUSTMENT
    
    if is_adult == "adult":
        weight_kg = max(150, min(700, weight_kg))
    else:
        weight_kg = max(80, min(400, weight_kg))
    
    breed_factors = {"beef": 1.0, "dairy": 0.95}
    weight_kg = weight_kg * breed_factors.get(breed_type, 1.0)
    weight_kg = float(round(weight_kg, 2))
    
    advices = get_farmer_advice(weight_kg, is_adult, breed_type)

    response = {
        "weight": weight_kg,
        "length_cm": L_cm,
        "girth_cm": G_cm,
        "advice": advices,
        "debug": {
            "w_px": w_px,
            "h_px": h_px,
            "depth_pixels": depth_pixels,
            "calibration_factor": float(round(CALIBRATION_FACTOR, 4)),
            "raw_weight": float(round(raw_weight, 2)),
            "final_adjustment": FINAL_ADJUSTMENT
        }
    }
    
    return convert_numpy_types(response)

@app.get("/", response_class=HTMLResponse)
async def index():
    return "<h1>SmartCow API is Online</h1><p>Use /calculate for POST requests.</p>"

@app.post("/calculate")
async def calculate_weight(
    file: UploadFile = File(...), 
    breed: str = Form(...), 
    age: str = Form(...), 
    strict: bool = Form(False)
):
    contents = await file.read()
    return process_weight_final_v4(contents, breed, age, strict)
