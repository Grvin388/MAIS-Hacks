import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

from squat import analyze_squat_video
from pushup import analyze_pushup_video

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

ALLOWED_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file field"}), 400

    exercise_type = request.form.get("exercise_type", "squat").lower()
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format '{ext}'. Use one of {ALLOWED_EXT}"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if exercise_type == "squat":
            result = analyze_squat_video(tmp_path)
        elif exercise_type == "pushup":
            result = analyze_pushup_video(tmp_path)
        else:
            result = {"error": f"Exercise '{exercise_type}' not supported. Try 'squat' or 'pushup'."}

        status = 200 if "error" not in result else 400
        return jsonify(result), status
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
