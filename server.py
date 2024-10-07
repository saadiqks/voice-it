from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import time
import threading
import magic
from voice_it import convert_file_to_mp3

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["FILE_EXPIRATION_TIME"] = 3600  # 1 hour in seconds

def validate_mime_type(file_path):
    mime = magic.Magic(mime=True)
    file_mime_type = mime.from_file(file_path)
    allowed_mime_types = ["text/plain", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/epub+zip"]
    return file_mime_type in allowed_mime_types

def delete_file_after_delay(file_path, delay):
    """Delete a file after a delay (in seconds)."""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted {file_path}")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "" or file.filename is None:
            return jsonify({"error": "No selected file"}), 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Validate MIME type
            if not validate_mime_type(file_path):
                os.remove(file_path) # Remove the file
                return jsonify({"error": "Invalid MIME type for the uploaded file."}), 400


            # Generate a timestamp and append it to the audio filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            audio_filename = os.path.splitext(filename)[0] + f"_{timestamp}.mp3"
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio_filename)
            convert_file_to_mp3(file_path, audio_path)

            # Schedule deletion of the original and audio file after an hour
            threading.Thread(target=delete_file_after_delay, args=(file_path, app.config["FILE_EXPIRATION_TIME"])).start()
            threading.Thread(target=delete_file_after_delay, args=(audio_path, app.config["FILE_EXPIRATION_TIME"])).start()

            return jsonify({"audio_file": audio_filename})

    return render_template("template.html")

@app.route("/downloads/<filename>")
def download_file(filename: str):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run()
