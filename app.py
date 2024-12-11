import os
import time

from flask import Flask, request, render_template, jsonify, send_file
from magic import Magic
from voice_it import convert_file_to_wav
from werkzeug.utils import secure_filename
from werkzeug.wrappers.response import Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def validate_mime_type(file_path: str) -> bool:
    mime = Magic(mime=True)
    file_mime_type = mime.from_file(file_path)
    allowed_mime_types = [
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]
    return file_mime_type in allowed_mime_types


@app.route("/", methods=["GET", "POST"])
def upload_file() -> tuple[Response, int] | Response | str:
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "" or file.filename is None:
            return jsonify({"error": "No selected file"}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("/tmp", filename)
            file.save(file_path)

            if not validate_mime_type(file_path):
                os.remove(file_path)
                return jsonify({"error": "Invalid MIME type for the uploaded file."}), 400

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            audio_filename = os.path.splitext(filename)[0] + f"_{timestamp}.wav"
            audio_path = os.path.join("/tmp", audio_filename)
            char_count = convert_file_to_wav(file_path, audio_path)

            if char_count != None:
                return jsonify({"error": f"File has too many characters: {char_count}"}), 400

            return jsonify({"audio_file": audio_filename})

    return render_template("template.html")


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_file(f"/tmp/{filename}", mimetype="audio/wav")


@app.route("/count/")
def count():
    char_count = 10
    seconds = char_count
    minutes = seconds // 60
    hours = minutes // 60
    estimated_time = None

    if seconds < 60:
        estimated_time = f"{seconds} second{'s' if seconds != 1 else ''}"
    elif minutes < 60:
        estimated_time = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        estimated_time = f"{hours} hour{'s' if hours != 1 else ''}"

    return Response(estimated_time)


if __name__ == "__main__":
    app.run()
