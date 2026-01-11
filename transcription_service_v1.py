import os
import time
# We switch to V1 (stable) because it reliably supports Diarization for LongRunningRecognize
from google.cloud import speech_v1p1beta1 as speech 
from google.cloud import secretmanager
from google import genai
from google.genai import types
from dotenv import load_dotenv
from secret_manager_utils import get_secret

load_dotenv()

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

def transcribe_gcs_file(gcs_uri):
    """
    Transcribes a long audio file from GCS using Speech-to-Text V1 Async API.
    Enables speaker diarization (2-6 speakers) and uses the 'latest_long' model.
    """
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)

    # V1 Diarization Configuration
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=6,
    )

    # Determine encoding based on file extension
    encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16 
    if gcs_uri.lower().endswith('.mp3'):
        encoding = speech.RecognitionConfig.AudioEncoding.MP3

    config = speech.RecognitionConfig(
        encoding=encoding, 
        language_code="en-US",
        model="latest_long", 
        diarization_config=diarization_config,
    )
    
    print(f"[DEBUG] Starting V1 LongRunningRecognize for {gcs_uri}...")
    operation = client.long_running_recognize(config=config, audio=audio)
    
    print(f"[DEBUG] Waiting for operation {operation.operation.name} to complete...")
    # Wait up to 3 hours (10800 seconds)
    response = operation.result(timeout=10800)

    # --- Transcript Reconstruction ---
    # V1 Diarization returns a list of words with speaker tags in the LAST result
    transcript_builder = []
    current_speaker = None
    current_sentence = []

    # In V1p1beta1 diarization, the words with tags are in the final result's alternative
    if response.results:
        # Get words from the last result which contains the full diarized word list
        result = response.results[-1]
        words_info = result.alternatives[0].words
        
        if words_info:
            for word_info in words_info:
                speaker = word_info.speaker_tag
                word = word_info.word

                if current_speaker is not None and speaker != current_speaker:
                    transcript_builder.append(f"**Speaker {current_speaker}:** {' '.join(current_sentence)}")
                    current_sentence = []
                
                current_speaker = speaker
                current_sentence.append(word)
            
            # Add final speaker block
            if current_sentence:
                transcript_builder.append(f"**Speaker {current_speaker}:** {' '.join(current_sentence)}")
        else:
            # Fallback if words_info is missing
            for res in response.results:
                transcript_builder.append(res.alternatives[0].transcript)

    full_transcript = "\n\n".join(transcript_builder)
    
    if not full_transcript.strip():
        full_transcript = "No transcript generated."

    return full_transcript


def refine_text_with_gemini(raw_text):
    """
    Refines raw transcript text using Gemini 3 Flash.
    Formats into clean paragraphs with speaker labels and fixed punctuation.
    """
    try:
        client = genai.Client(api_key=get_secret())
        
        system_instruction = (
        "You are an expert transcriber. You will receive a raw transcript with speaker labels. "
        "Format it into clean, readable paragraphs. Fix punctuation and capitalization. "
        "Do NOT summarize; keep every word. Differentiate speakers clearly (e.g., **Speaker 1:**). "
        "Use <br> for new lines to ensure proper HTML rendering."
    )

        print(f"[DEBUG] Sending transcript (length: {len(raw_text)}) to Gemini 3 Flash...")
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
    except Exception as e:
        print(f"[ERROR] Gemini API Call failed: {str(e)}")
        # Return raw text so the user doesn't lose the transcription if refinement fails
        return f"Refinement failed. Raw Transcript:\n\n{raw_text}"