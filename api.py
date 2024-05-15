from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
import cv2
import numpy as np
from detector import predict
import face_recognition

app = Flask(__name__)

SUPPORTED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/bmp', 'image/gif'}

@app.route('/predict/emotion', methods=['POST'])
def image_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.content_type in SUPPORTED_IMAGE_TYPES:
        try:
            file_content = file.read()
            i_buffer = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(i_buffer, cv2.IMREAD_COLOR)

            if image is None:
                return jsonify({"error": "Could not decode the image. Ensure it is in a supported format."}), 422

            result = predict(image)
            if result is None:
                return jsonify({"error": "No face detected"}), 404

            emotion, (x, y, w, h) = result
            return jsonify({"emotion": emotion, "x": x, "y": y, "w": w, "h": h}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Unsupported image type"}), 415

# In-memory storage for face encodings and names
face_db = {}

@app.route('/register', methods=['POST'])
def register_face():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "No image or name provided"}), 400

    file = request.files['image']
    name = request.form['name']
    image = face_recognition.load_image_file(file)
    
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings) > 0:
        face_db[name] = face_encodings[0].tolist()
        return jsonify({"status": "Face registered"}), 200
    else:
        return jsonify({"error": "No face detected"}), 400

@app.route('/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = face_recognition.load_image_file(file)
    
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([np.array(encoding) for encoding in face_db.values()], face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = list(face_db.keys())[first_match_index]
        names.append(name)

    return jsonify({"names": names})

@app.route('/')
def read_root():
    return jsonify({"Hello": "World"})

if __name__ == '__main__':
    app.run(debug=True)
