import os
import uuid
import threading
from flask import Flask, request, render_template, jsonify, send_from_directory
from google.cloud import storage
from transcription_service import transcribe_gcs_file, refine_text_with_gemini
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

app = Flask(__name__)

# Configuration
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')

# Simple in-memory job store (for demo purposes)
jobs = {}

def upload_to_gcs(file_storage, bucket_name):
    """Uploads a file to Google Cloud Storage and returns the gs:// URI."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"uploads/{uuid.uuid4()}-{file_storage.filename}"
    blob = bucket.blob(blob_name)
    
    # Upload from stream
    blob.upload_from_file(file_storage)
    
    return f"gs://{bucket_name}/{blob_name}"

def run_transcription_pipeline(job_id, gcs_uri):
    """Background task to process audio transcription."""
    try:
        jobs[job_id]['status'] = 'transcribing'
        raw_text = transcribe_gcs_file(gcs_uri)
        
        jobs[job_id]['status'] = 'refining'
        refined_text = refine_text_with_gemini(raw_text)
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = refined_text
    except Exception as e:
        print(f"Error in transcription pipeline: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not GCS_BUCKET_NAME:
        return jsonify({"error": "GCS_BUCKET_NAME not configured"}), 500

    try:
        # Step 1: Upload to GCS
        gcs_uri = upload_to_gcs(file, GCS_BUCKET_NAME)
        
        # Step 2: Create a Job ID
        job_id = str(uuid.uuid4())
        jobs[job_id] = {'status': 'pending', 'uri': gcs_uri}
        
        # Step 3: Trigger background processing
        thread = threading.Thread(target=run_transcription_pipeline, args=(job_id, gcs_uri))
        thread.start()
        
        return jsonify({"job_id": job_id, "gcs_uri": gcs_uri})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
