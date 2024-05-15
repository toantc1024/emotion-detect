from detector import predict
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import cv2
import numpy as np

app =FastAPI() 


# initialize FastAPI
@app.post("/predict/emotion")
async def image_upload(
        file: UploadFile = File(..., description="Uploaded File")):
    content_type = file.content_type
    if content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=422, detail={"error": str("Invalid content type")})

    file_content = await file.read()
    i_buffer = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(i_buffer, cv2.IMREAD_COLOR)
    # Predict
    result = predict(image)
    if result is None:
        return {"error": "No face detected"}
    emotion, (x, y, w, h) = result
    return {"emotion": emotion, "x": x, "y": y, "w": w, "h": h}
@app.get("/")
async def read_root():
    return {"Hello": "World"}
