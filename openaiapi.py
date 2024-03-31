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
