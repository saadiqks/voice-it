import os
import time
import zipfile

from flask import Flask, request, render_template, jsonify, send_file
from magic import Magic
from voice_it import convert_file_to_wav
from werkzeug.utils import secure_filename
from werkzeug.wrappers.response import Response

app = Flask(__name__)

def validate_mime_type(file_path: str) -> bool:
    mime = Magic(mime=True)
    file_mime_type = mime.from_file(file_path)
    allowed_mime_types = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
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
            convert_file_to_wav(file_path, audio_path)

            # Zip the audio file
            zip_filename = audio_filename.replace(".wav", ".zip")
            zip_path = os.path.join("/tmp", zip_filename)
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(audio_path, arcname=audio_filename)

            return jsonify({"audio_file": zip_filename})

    return render_template("template.html")


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    return send_file(f"/tmp/{filename}", mimetype="application/zip")


if __name__ == "__main__":
    app.run()
