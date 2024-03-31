from fastapi import FastAPI, File, Response
from fastapi.responses import FileResponse
import os
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import subprocess

app = FastAPI()

@app.get("/process/{name}")
async def process_audio(name: str):
    command = f"python inference.py --driven_audio /tmp/{name}.wav --source_image demo/wangpeng.png --enhancer gfpgan --output {name}"
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    print("output:",output)
    print("error:",error)

    if process.returncode == 0:
        return PlainTextResponse(content="Process completed successfully")
    else:
        return PlainTextResponse(content=f"Error: {error.decode()}", status_code=500)


@app.get("/download_video/{video_name}")
async def download_video(video_name: str):
    video_path = f"./results/{video_name}"+".mp4"

    if os.path.exists(video_path):
        return FileResponse(video_path, media_type="video/mp4", filename=video_name)
    else:
        return Response(content="Video not found", status_code=404)