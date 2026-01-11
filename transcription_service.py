import os
import time
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech
from google import genai
from google.genai import types
from dotenv import load_dotenv
from secret_manager_utils import get_secret

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "global"  # Must be global for Diarization in BatchRecognize

def transcribe_gcs_file(gcs_uri):
    """
    Transcribes a long audio file from GCS using Speech-to-Text V2 Batch API.
    Enables speaker diarization (2-6 speakers) and uses the 'chirp_3' model.
    """
    client = speech_v2.SpeechClient() # Global location doesn't need regional endpoint

    # Recognizer configuration
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        features=cloud_speech.RecognitionFeatures(
            diarization_config=cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=2,
                max_speaker_count=6,
            ),
        ),
        model="chirp_3",
        language_codes=["en-US"],
    )

    output_config = cloud_speech.RecognitionOutputConfig(
        inline_response_config=cloud_speech.InlineOutputConfig(),
    )

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
        config=config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=output_config,
    )

    operation = client.batch_recognize(request=request)
    print(f"Waiting for transcription operation to complete: {operation.operation.name}")
    
    response = operation.result(timeout=3600)  # Wait up to 1 hour
    
    # Process results from inline destination
    transcript = ""
    for result in response.results[gcs_uri].transcript.results:
        if result.alternatives:
            # Note: For diarization, we might need to process words or specific fields.
            # Speech V2 Batch diarization results are usually embedded in the alternatives.
            transcript += result.alternatives[0].transcript + "\n"
            
    return transcript

def refine_text_with_gemini(raw_text):
    """
    Refines raw transcript text using Gemini 3 Flash.
    Formats into clean paragraphs with speaker labels and fixed punctuation.
    """
    api_key = get_secret("GEMINI_API_KEY_ID")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") # Fallback to ENV for dev
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY could not be retrieved")

    client = genai.Client(api_key=api_key)
    
    system_instruction = (
        "You are an expert transcriber. You will receive a raw transcript with speaker labels. "
        "Format it into clean, readable paragraphs. Fix punctuation and capitalization. "
        "Do NOT summarize; keep every word. Differentiate speakers clearly (e.g., **Speaker 1:**). "
        "Use <br> for new lines to ensure proper HTML rendering."
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview", # As requested in the project plan
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Refine this transcript:\n\n{raw_text}")]
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3
        )
    )
    
    return response.text
