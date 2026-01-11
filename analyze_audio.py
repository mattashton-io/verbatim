from google.cloud import storage
import io
from pydub.utils import mediainfo
import os
import sys

# Usage: python analyze_audio.py gs://bucket/path/to/file.mp3

def analyze_gcs_audio(gcs_uri):
    print(f"Analyzing {gcs_uri}...")
    
    try:
        storage_client = storage.Client()
        
        # Parse URI
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download to temp file (mediainfo needs a file path usually, or we can try headers)
        # downloading header check or full file? MP3 header is enough but let's download small chunk or full.
        # mediainfo works best on file.
        
        filename = "temp_debug_audio" + os.path.splitext(blob_name)[1]
        blob.download_to_filename(filename)
        
        info = mediainfo(filename)
        print("--- Audio Metadata ---")
        print(f"Format: {info.get('format_name')}")
        print(f"Sample Rate: {info.get('sample_rate')} Hz")
        print(f"Channels: {info.get('channels')}")
        print(f"Bit Rate: {info.get('bit_rate')}")
        print(f"Codec: {info.get('codec_name')}")
        print("----------------------")
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
            
    except Exception as e:
        print(f"Error analyzing audio: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a GCS URI.")
        sys.exit(1)
    
    analyze_gcs_audio(sys.argv[1])
