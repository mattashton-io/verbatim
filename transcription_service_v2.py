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
# MUST be "global" because Batch API Diarization is not supported in regional endpoints yet.
LOCATION = "global" 

def transcribe_gcs_file(gcs_uri):
    """
    Transcribes a long audio file from GCS using Speech-to-Text V2 Batch API.
    Enables speaker diarization (2-6 speakers) and uses the 'long' model.
    """
    # Instantiate client without ClientOptions (defaults to global endpoint)
    client = speech_v2.SpeechClient()

    # Recognizer configuration
    # Use 'long' model which supports Diarization globally for Batch requests.
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        features=cloud_speech.RecognitionFeatures(
            diarization_config=cloud_speech.SpeakerDiarizationConfig(
                min_speaker_count=2,
                max_speaker_count=6,
            ),
        ),
        model="long", 
        language_codes=["en-US"],
    )

    output_config = cloud_speech.RecognitionOutputConfig(
        inline_response_config=cloud_speech.InlineOutputConfig(),
    )

    # Create the BatchRecognizeRequest
    # Note: "recognizers/_" means we are providing the config inline rather than referencing a pre-created recognizer resource.
    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
        config=config,
        files=[cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)],
        recognition_output_config=output_config,
    )

    # Create the batch recognition operation
    operation = client.batch_recognize(request=request)
    print(f"Waiting for transcription operation to complete: {operation.operation.name}")
    
    # Wait for the operation to complete (timeout set to 3600s = 1 hour)
    response = operation.result(timeout=3600)
    
    # Process results from inline destination
    transcript = ""
    
    # The response results are keyed by the input GCS URI
    if gcs_uri in response.results:
        file_result = response.results[gcs_uri]
        
        # Check for errors in the specific file result
        if file_result.error and file_result.error.message:
            raise Exception(f"Transcription failed: {file_result.error.message}")

        # Concatenate transcript segments
        if file_result.transcript and file_result.transcript.results:
            for result in file_result.transcript.results:
                if result.alternatives:
                    # Append the transcript from the first alternative
                    transcript += result.alternatives[0].transcript + "\n"
    else:
        raise Exception("Transcription result not found for the provided URI.")
            
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
        model="gemini-3-flash-preview", 
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