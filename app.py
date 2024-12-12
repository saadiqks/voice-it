import os
import time

from flask import Flask, request, render_template, jsonify, send_file
from magic import Magic
from voice_it import get_text, convert_file_to_wav
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


def save_uploaded_file(file):
    """
    Saves the uploaded file and returns the file path.
    Performs basic validations.
    """
    if not file or file.filename is None or file.filename == "":
        return None

    filename = secure_filename(file.filename)
    file_path = os.path.join("/tmp", filename)
    file.save(file_path)

    # Validate MIME type
    if not validate_mime_type(file_path):
        os.remove(file_path)
        return None

    return file_path


@app.route("/", methods=["GET", "POST"])
def upload_file() -> tuple[Response, int] | Response | str:
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        file_path = save_uploaded_file(file)

        if file_path is None:
            return jsonify({"error": "Invalid file or MIME type"}), 400

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        audio_filename = os.path.splitext(os.path.basename(file_path))[0] + f"_{timestamp}.wav"
        audio_path = os.path.join("/tmp", audio_filename)
        char_count = convert_file_to_wav(file_path, audio_path)

        if char_count is not None:
            return jsonify({"error": f"File has too many characters: {char_count}"}), 400

        return jsonify({"audio_file": audio_filename})

    return render_template("template.html")


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename: str) -> Response:
    return send_file(f"/tmp/{filename}", mimetype="audio/wav")


@app.route("/count/", methods=["POST"])
def count() -> Response:
    if "file" not in request.files:
        return Response(str(0))

    file = request.files["file"]
    file_path = save_uploaded_file(file)

    if file_path is None:
        return Response(str(0))

    text = get_text(file_path)
    return Response(str(len(text) // 100))

if __name__ == "__main__":
    app.run()
