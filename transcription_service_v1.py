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
    # V1 API is strict: MP3 files must use MP3 encoding, WAV usually uses LINEAR16
    encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16 # Default to WAV/PCM
    if gcs_uri.lower().endswith('.mp3'):
        encoding = speech.RecognitionConfig.AudioEncoding.MP3

    config = speech.RecognitionConfig(
        # FIX: Enum members must be UPPERCASE (Linear16 -> LINEAR16)
        encoding=encoding, 
        language_code="en-US",
        model="latest_long", # Optimized for long audio
        diarization_config=diarization_config,
    )
    
    print(f"Starting V1 LongRunningRecognize for {gcs_uri} with encoding {encoding}...")
    operation = client.long_running_recognize(config=config, audio=audio)
    
    print(f"Waiting for operation {operation.operation.name} to complete...")
    # Wait up to 3 hours (10800 seconds)
    response = operation.result(timeout=10800)

    # --- Transcript Reconstruction ---
    # V1 Diarization returns words with speaker tags. We must group them.
    
    transcript_builder = []
    current_speaker = None
    current_sentence = []

    # Iterate through all results to gather words
    for result in response.results:
        # Taking the top alternative
        alternative = result.alternatives[0]
        
        # Check if words are present (needed for diarization)
        if alternative.words:
            for word_info in alternative.words:
                speaker = word_info.speaker_tag
                word = word_info.word

                # If speaker changes, push the previous sentence and start new
                if current_speaker is not None and speaker != current_speaker:
                    transcript_builder.append(f"**Speaker {current_speaker}:** {' '.join(current_sentence)}")
                    current_sentence = []
                
                current_speaker = speaker
                current_sentence.append(word)
        else:
            # Fallback if no word-level info (unlikely with diarization enabled)
            transcript_builder.append(alternative.transcript)

    # Append the final sentence
    if current_speaker is not None and current_sentence:
         transcript_builder.append(f"**Speaker {current_speaker}:** {' '.join(current_sentence)}")
    
    full_transcript = "\n\n".join(transcript_builder)
    
    # Fallback: If diarization didn't tag speakers (sometimes happens with poor audio),
    # just grab the plain text.
    if not full_transcript.strip():
        full_transcript = "\n".join([result.alternatives[0].transcript for result in response.results])

    return full_transcript


def refine_text_with_gemini(raw_text):
    """
    Refines raw transcript text using Gemini 3 Flash.
    Formats into clean paragraphs with speaker labels and fixed punctuation.
    """
    try:
        # Create the Secret Manager client.
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version.
        name = "projects/396631018769/secrets/verbatim-gemini/versions/latest"

        # Access the secret version.
        response = client.access_secret_version(request={"name": name})

        # Extract the payload.
        secret_string = response.payload.data.decode("UTF-8")

        genai.configure(api_key=secret_string)
        
        system_instruction = (
        "You are an expert transcriber. You will receive a raw transcript with speaker labels. "
        "Format it into clean, readable paragraphs. Fix punctuation and capitalization. "
        "Do NOT summarize; keep every word. Differentiate speakers clearly (e.g., **Speaker 1:**). "
        "Use <br> for new lines to ensure proper HTML rendering."
    )

        response = client.models.generate_content(
          model="gemini-2.5-flash", 
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
        print(f"Error refining text with Gemini: {e}")
        return raw_text # Return original text if Gemini fails