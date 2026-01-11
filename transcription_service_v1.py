import os
import time
import uuid
import json
import subprocess
from google.cloud import speech_v1p1beta1 as speech 
from google.cloud import secretmanager
from google.cloud import storage
from google import genai
from google.genai import types
from dotenv import load_dotenv
from secret_manager_utils import get_secret

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

def convert_to_flac_mono(gcs_uri):
    """
    Downloads audio from GCS, converts to FLAC mono 16kHz using ffmpeg subprocess, 
    uploads back to GCS, and returns the new GCS URI.
    """
    print(f"[DEBUG] converting {gcs_uri} to FLAC mono 16kHz using ffmpeg...")
    storage_client = storage.Client()
    
    # Parse URI
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download to temp
    local_filename = f"/tmp/{uuid.uuid4()}_{os.path.basename(blob_name)}"
    blob.download_to_filename(local_filename)
    
    try:
        output_filename = local_filename + ".flac"
        
        # Convert using ffmpeg directly (no pydub/audioop dependency)
        # -i input
        # -ac 1 (mono)
        # -ar 16000 (16kHz sample rate)
        # -y (overwrite output)
        command = [
            "ffmpeg", 
            "-i", local_filename, 
            "-ac", "1", 
            "-ar", "16000", 
            "-y", 
            output_filename
        ]
        
        print(f"[DEBUG] Running conversion: {' '.join(command)}")
        # Run ffmpeg, capture output to avoid cluttering logs unless error
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"[ERROR] ffmpeg failed: {result.stderr}")
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
            
        # Upload converted
        converted_blob_name = f"converted/{uuid.uuid4()}.flac"
        converted_blob = bucket.blob(converted_blob_name)
        converted_blob.upload_from_filename(output_filename)
        
        print(f"[DEBUG] Converted and uploaded to gs://{bucket_name}/{converted_blob_name}")
        
        # Cleanup
        if os.path.exists(local_filename): os.remove(local_filename)
        if os.path.exists(output_filename): os.remove(output_filename)
        
        return f"gs://{bucket_name}/{converted_blob_name}"
        
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        # Try cleanup
        if os.path.exists(local_filename): os.remove(local_filename)
        if 'output_filename' in locals() and os.path.exists(output_filename): os.remove(output_filename)
        raise e

def transcribe_gcs_file(gcs_uri):
    """
    Transcribes a long audio file from GCS using Speech-to-Text V1 Async API.
    Converts audio to FLAC mono 16kHz first to ensure compatibility.
    """
    # Step 1: Convert Audio
    try:
        converted_gcs_uri = convert_to_flac_mono(gcs_uri)
    except Exception as e:
        return f"Error converting audio: {e}"

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=converted_gcs_uri)

    # V1 Diarization Configuration
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=6,
    )

    # Config for FLAC 16k Mono
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        model="latest_long", # Safe to use latest_long with clean FLAC
        diarization_config=diarization_config,
        enable_automatic_punctuation=True,
    )

    # Output Config to write results to GCS
    output_gcs_uri_str = converted_gcs_uri.replace("gs://", "").split('/', 1)
    bucket_name = output_gcs_uri_str[0]
    output_filename = f"transcripts/{int(time.time())}.json"
    output_uri = f"gs://{bucket_name}/{output_filename}"

    output_config = speech.TranscriptOutputConfig(gcs_uri=output_uri)

    print(f"[DEBUG] Starting V1 LongRunningRecognize for {converted_gcs_uri}...")
    
    request = speech.LongRunningRecognizeRequest(
        config=config,
        audio=audio,
        output_config=output_config
    )

    operation = client.long_running_recognize(request=request)
    
    print(f"[DEBUG] Waiting for operation {operation.operation.name} to complete...")
    operation.result(timeout=10800)
    
    print(f"[DEBUG] Operation complete. Fetching results from {output_uri}...")
    
    # Read result from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(output_filename)
    
    try:
        result_json_bytes = blob.download_as_bytes()
        result_data = json.loads(result_json_bytes)
    except Exception as e:
        print(f"[ERROR] Failed to download/parse transcript: {e}")
        return "Transcription failed: could not retrieve results."
    
    results_list = result_data.get('results', [])
    print(f"[DEBUG] Processing {len(results_list)} results for diarization...")
    
    transcript_builder = []
    current_speaker = None
    current_sentence = []
    
    found_diarization = False
    
    for i, result in enumerate(results_list):
        alternatives = result.get('alternatives', [])
        if not alternatives:
            continue
            
        alternative = alternatives[0]
        words = alternative.get('words', [])
        transcript = alternative.get('transcript', '')
        
        if i < 3:
             # Debugging first few words
             if words:
                 print(f"[DEBUG] Result {i} first word: {words[0].get('word')} (Speaker: {words[0].get('speakerTag')})")
        
        if words:
            found_diarization = True
            for word_info in words:
                speaker = word_info.get('speakerTag') 
                word = word_info.get('word')

                if current_speaker is not None and speaker != current_speaker:
                    transcript_builder.append(f"**Speaker {current_speaker}:** {' '.join(current_sentence)}")
                    current_sentence = []
                
                current_speaker = speaker
                current_sentence.append(word)
        else:
             if transcript:
                transcript_builder.append(transcript)

    # Add the final speaker block
    if current_sentence:
         transcript_builder.append(f"**Speaker {current_speaker}:** {' '.join(current_sentence)}")
    
    full_transcript = "\n\n".join(transcript_builder)
    
    if not full_transcript:
        print("[DEBUG] Warning: full_transcript is empty. No text produced by STT.")
        full_transcript = "No transcript generated."
    else:
        print(f"[DEBUG] Reconstructed transcript length: {len(full_transcript)} chars. Diarization found: {found_diarization}")

    return full_transcript


def refine_text_with_gemini(raw_text):
    """
    Refines raw transcript text using Gemini 3 Flash.
    Formats into clean paragraphs with speaker labels and fixed punctuation.
    """
    if not raw_text or raw_text == "No transcript generated." or raw_text.startswith("Error"):
        print("[DEBUG] No valid text found to refine. Skipping Gemini call.")
        return raw_text

    print("[DEBUG] Attempting to retrieve Gemini API Key...")
    try:
        client = genai.Client(api_key=get_secret())
        
        system_instruction = (
        "You are a professional meeting scribe and editor. Your task is to take the provided raw "
        "transcript and format it into clean, readable paragraphs. You MUST preserve every word spoken. "
        "1. Fix punctuation and capitalization.\n"
        "2. Ensure speaker labels are clear (e.g., **Speaker 1:**).\n"
        "3. Use <br> tags for all line breaks to support web rendering.\n"
        "4. Do NOT summarize or remove content."
    )

        preview = (raw_text[:100] + '...') if len(raw_text) > 100 else raw_text
        print(f"[DEBUG] Sending transcript to Gemini 3 Flash. Preview: {preview}")

        response = client.models.generate_content(
            model="gemini-3-flash-preview", 
            contents=f"Refine this transcript:\n\n{raw_text}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )

        print("[DEBUG] Gemini refinement complete.")   
        return response.text

    except Exception as e:
        print(f"[ERROR] Gemini API Call failed: {str(e)}")
        return f"Refinement failed. Raw Transcript:\n\n{raw_text}"