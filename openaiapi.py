from fastapi import FastAPI, File, Response
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI,BackgroundTasks
from fastapi.responses import PlainTextResponse
import subprocess
import asyncio
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

async def run_process_async(command):
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout, stderr

@app.get("/process/{name}")
async def process_audio(name: str, background_tasks: BackgroundTasks):
    if not os.path.exists(f"/tmp/{name}.wav"):
        return PlainTextResponse(content=f"Error: file not exist!", status_code=500)

    command = f"python inference.py --driven_audio /tmp/{name}.wav --source_image demo/wangpeng.png --enhancer gfpgan --output {name}"
    print(command)

    background_tasks.add_task(run_process_async, command)

    return PlainTextResponse(content="Processing started in the background.")


@app.get("/video/{video_name}")
async def download_video(video_name: str):
    video_path = f"./results/{video_name}"+".mp4"

    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4", filename=video_name+".mp4")
    else:
        return Response(content="Video not found", status_code=404)