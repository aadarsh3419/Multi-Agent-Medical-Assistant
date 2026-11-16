import asyncio
import os
import shutil
import uuid
import tempfile
from pathlib import Path
from typing import Any, Dict, Union, Optional, List
import glob
import threading
import time
import traceback
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, Cookie
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
import requests
from werkzeug.utils import secure_filename

if os.name == "nt" and not shutil.which("ffmpeg"):
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        ffmpeg_binaries = winget_packages.glob(
            "Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"
        )
        ffmpeg_path = next((path for path in ffmpeg_binaries if path.is_file()), None)
        if ffmpeg_path:
            os.environ["PATH"] = (
                str(ffmpeg_path.parent) + os.pathsep + os.environ.get("PATH", "")
            )

from pydub import AudioSegment
from elevenlabs.client import ElevenLabs

from config import Config
from agents.agent_decision import process_query


config = Config()


app = FastAPI(title="Multi-Agent Medical Chatbot", version="2.0")


UPLOAD_FOLDER = "uploads/backend"
FRONTEND_UPLOAD_FOLDER = "uploads/frontend"
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
SPEECH_DIR = "uploads/speech"
pending_validations: dict[str, str] = {}


for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT, SPEECH_DIR]:
    os.makedirs(directory, exist_ok=True)


app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


templates = Jinja2Templates(directory="templates")


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_audio():
    """Deletes all .mp3 files in the uploads/speech folder every 5 minutes."""
    while True:
        try:
            files = glob.glob(f"{SPEECH_DIR}/*.mp3")
            for file in files:
                os.remove(file)
            print("Cleaned up old speech files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        time.sleep(300)


cleanup_thread = threading.Thread(target=cleanup_old_audio, daemon=True)
cleanup_thread.start()

class QueryRequest(BaseModel):
    query: str
    conversation_history: List = []

class SpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None


async def run_agent_query(query: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(process_query, query),
            timeout=27,
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The request took too long. Please try again.",
        ) from exc
    except Exception as exc:
        traceback.print_exc()
        if "429" in str(exc) or "quota" in str(exc).lower():
            raise HTTPException(
                status_code=503,
                detail="Gemini request quota is unavailable. Check your Google AI quota and try again later.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="The AI service could not complete this request.",
        ) from exc


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}

@app.post("/chat")
async def chat(
    request: QueryRequest,
    response: Response,
    session_id: Optional[str] = Cookie(None)
):
    """Process user text query through the multi-agent system."""

    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        response_data = await run_agent_query(request.query)
        response_text = response_data['messages'][-1].content


        response.set_cookie(key="session_id", value=session_id)


        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data["agent_name"]
        }


        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(
    response: Response,
    image: UploadFile = File(...),
    text: str = Form(""),
    session_id: Optional[str] = Cookie(None)
):
    """Process medical image uploads with optional text input."""

    if not allowed_file(image.filename):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "agent": "System",
                "response": "Unsupported file type. Allowed formats: PNG, JPG, JPEG"
            }
        )


    file_content = await image.read()
    if len(file_content) > config.api.max_image_upload_size * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={
                "status": "error",
                "agent": "System",
                "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
            }
        )


    if not session_id:
        session_id = str(uuid.uuid4())


    filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        query = {"text": text, "image": file_path}
        response_data = await run_agent_query(query)
        response_text = response_data['messages'][-1].content


        response.set_cookie(key="session_id", value=session_id)


        result = {
            "status": "success",
            "response": response_text,
            "agent": response_data["agent_name"]
        }
        if response_data["agent_name"].endswith(", HUMAN_VALIDATION"):
            pending_validations[session_id] = response_text


        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
            else:
                print("Skin Lesion Output path does not exist.")

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError as cleanup_error:
                print(f"Failed to remove temporary upload: {type(cleanup_error).__name__}")

@app.post("/validate")
def validate_medical_output(
    response: Response,
    validation_result: str = Form(...),
    comments: Optional[str] = Form(None),
    session_id: Optional[str] = Cookie(None)
):
    """Handle human validation for medical AI outputs."""

    if not session_id:
        session_id = str(uuid.uuid4())

    try:

        response.set_cookie(key="session_id", value=session_id)

        pending_response = pending_validations.pop(session_id, None)
        if pending_response is None:
            raise HTTPException(
                status_code=409,
                detail="There is no pending medical result to validate.",
            )

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": pending_response
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": pending_response
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Endpoint to transcribe speech using ElevenLabs API"""
    if not config.speech.eleven_labs_api_key:
        return JSONResponse(
            status_code=503,
            content={"error": "Speech recognition is not configured. Check the ElevenLabs API key."}
        )
    if not audio.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "No audio file selected"}
        )

    try:

        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.webm"


        audio_content = await audio.read()
        with open(temp_audio, "wb") as f:
            f.write(audio_content)


        file_size = os.path.getsize(temp_audio)
        print(f"Received audio file size: {file_size} bytes")

        if file_size == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Received empty audio file"}
            )


        mp3_path = f"./{SPEECH_DIR}/speech_{uuid.uuid4()}.mp3"

        try:

            audio = AudioSegment.from_file(temp_audio)
            audio.export(mp3_path, format="mp3")


            mp3_size = os.path.getsize(mp3_path)
            print(f"Converted MP3 file size: {mp3_size} bytes")

            with open(mp3_path, "rb") as mp3_file:
                audio_data = mp3_file.read()
            print(f"Converted audio file into byte array successfully!")

            speech_client = ElevenLabs(
                api_key=config.speech.eleven_labs_api_key,
                timeout=15,
            )
            transcription = speech_client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="eng",
                diarize=True,
            )


            try:
                os.remove(temp_audio)
                os.remove(mp3_path)
                print(f"Deleted temp files: {temp_audio}, {mp3_path}")
            except Exception as e:
                print(f"Could not delete file: {e}")

            if transcription.text:
                return {"transcript": transcription.text}
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": f"API error: {transcription}", "details": transcription.text}
                )

        except Exception as e:
            traceback.print_exc()
            print(f"Speech recognition failed: {type(e).__name__}")
            return JSONResponse(
                status_code=503,
                content={"error": "Speech recognition is unavailable. Check the ElevenLabs key or quota."}
            )

    except Exception as e:
        traceback.print_exc()
        print(f"Speech transcription failed: {type(e).__name__}")
        return JSONResponse(
            status_code=503,
            content={"error": "Speech recognition is unavailable. Check the ElevenLabs key or quota."}
        )

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Endpoint to generate speech using ElevenLabs API"""
    if not config.speech.eleven_labs_api_key:
        return JSONResponse(
            status_code=503,
            content={"error": "Voice generation is not configured. Check the ElevenLabs API key."}
        )
    try:
        text = request.text
        selected_voice_id = request.voice_id or config.speech.eleven_labs_voice_id

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required"}
            )


        elevenlabs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}/stream"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config.speech.eleven_labs_api_key
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }


        response = requests.post(
            elevenlabs_url,
            headers=headers,
            json=payload,
            timeout=(5, 15),
        )

        if response.status_code != 200:
            return JSONResponse(
                status_code=503,
                content={"error": "Voice generation is unavailable. Check the ElevenLabs key or quota."}
            )


        os.makedirs(SPEECH_DIR, exist_ok=True)
        temp_audio_path = f"./{SPEECH_DIR}/{uuid.uuid4()}.mp3"
        with open(temp_audio_path, "wb") as f:
            f.write(response.content)


        return FileResponse(
            path=temp_audio_path,
            media_type="audio/mpeg",
            filename="generated_speech.mp3"
        )

    except Exception as e:
        if isinstance(e, requests.RequestException):
            traceback.print_exc()
            return JSONResponse(
                status_code=503,
                content={"error": "Voice generation is unavailable. Check the ElevenLabs key or quota."}
            )
        raise HTTPException(status_code=500, detail="Voice generation failed.") from e


@app.exception_handler(413)
async def request_entity_too_large(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "status": "error",
            "agent": "System",
            "response": f"File too large. Maximum size allowed: {config.api.max_image_upload_size}MB"
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host=config.api.host, port=config.api.port)