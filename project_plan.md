# **Project Plan: Verbatim**

**Project Goal:** Build an asynchronous web app capable of ingesting 3+ hours of audio, transcribing it using Google Cloud Speech-to-Text V2 (Batch API), and formatting the output using Gemini 3 Flash.

## **Phase 1: Infrastructure & Scaffolding**

Prompt:  
I need to build a Flask web app for long-form speech-to-text.  
**Context:**

* Reference my existing virtual-echo-chamber repo for the file structure (app.py, templates/index.html, static/) and Tailwind CSS styling (dark mode).  
* The app needs to handle large audio uploads (3+ hours).

**Task:**

1. Scaffold a new Flask app in app.py.  
2. Create a route /upload that accepts .m4a, .mp3, and .wav files.  
3. Instead of saving locally, write a function to upload these files directly to a Google Cloud Storage bucket defined by os.environ.get('GCS\_BUCKET\_NAME').  
4. Return the gs:// URI after upload.  
5. Update requirements.txt to include google-cloud-storage, google-cloud-speech, and google-genai.  
6. Create a templates/index.html with a file upload form that supports a loading state.

## **Phase 2: The Logic Core (STT V2 & Gemini 3\)**

Prompt:  
I need the backend logic for processing the audio. Create a new file transcription\_service.py with two main functions.  
**Function 1: transcribe\_gcs\_file(gcs\_uri)**

* Use the google-cloud-speech library (v2).  
* Create a **BatchRecognizeRequest** (this is critical for long files).  
* Config: Enable Speaker Diarization (min 2, max 6 speakers), use the 'long' model.  
* Wait for the operation to complete and return the raw transcript text with speaker labels.

**Function 2: refine\_text\_with\_gemini(raw\_text)**

* Use the google-genai library (or google.generativeai if easier) to call the **Gemini 3 Flash** model.  
* **Model Name:** Use gemini-3-flash (or the latest available alias).  
* **System Prompt:** 'You are an expert transcriber. You will receive a raw transcript with speaker labels. Format it into clean, readable paragraphs. Fix punctuation and capitalization. Do NOT summarize; keep every word. Differentiate speakers clearly (e.g., **Speaker 1:**).'

Wire these functions into app.py so that when a file is uploaded, it triggers this pipeline in a background thread and returns a Job ID.

## **Phase 3: The Frontend Polling**

Prompt:  
Update index.html to handle the asynchronous nature of this app.

1. Use JavaScript fetch for the form submission.  
2. When the user clicks 'Transcribe', show a progress bar that says 'Uploading & Processing...'.  
3. The backend /upload endpoint should return a job\_id.  
4. Write a JavaScript polling function that checks /status/\<job\_id\> every 5 seconds.  
5. When the status is 'completed', display the formatted text in a large text area and offer a 'Download' button.  
6. Keep the dark/light mode toggle from the reference code.

## **Technical Context Snippet**

**Note:** If the agent struggles with the specific syntax for Gemini 3, paste this snippet into the chat to guide it.

\# CONTEXT: How to use Gemini 3 Flash for this app  
import os  
from google import genai  
from google.genai import types

def refine\_with\_gemini\_3(raw\_text):  
    client \= genai.Client(api\_key=os.environ\["GEMINI\_API\_KEY"\])  
      
    response \= client.models.generate\_content(  
        model="gemini-3-flash",  \# Ensure this matches your specific available model version  
        contents=\[  
            types.Content(  
                role="user",  
                parts=\[  
                    types.Part.from\_text(text=f"Refine this transcript:\\n\\n{raw\_text}")  
                \]  
            )  
        \],  
        config=types.GenerateContentConfig(  
            system\_instruction="Format into clear paragraphs with speaker labels. Fix punctuation.",  
            temperature=0.3  
        )  
    )  
    return response.text  