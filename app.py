import os
import time

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from werkzeug.wrappers.response import Response

from config import FileConfig
from conversion_service import convert_file_to_wav, save_uploaded_file
from text_service import get_text

app = Flask(__name__)
CORS(app)


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
        audio_filename = os.path.splitext(os.path.basename(file_path))[0] + f"_{timestamp}.mp3"
        audio_path = os.path.join(FileConfig.UPLOAD_FOLDER, audio_filename)
        char_count = convert_file_to_wav(file_path, audio_path)

        if char_count is not None:
            return jsonify({"error": f"File has too many characters: {char_count}"}), 400

        return jsonify({"audio_file": audio_filename})

    return render_template("template.html")


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename: str) -> Response:
    return send_file(f"{FileConfig.UPLOAD_FOLDER}/{filename}", mimetype="audio/mp3")


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
