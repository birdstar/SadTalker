from fastapi import FastAPI, File, Response, UploadFile
from fastapi.responses import FileResponse,StreamingResponse
import os
from fastapi import FastAPI,BackgroundTasks
from fastapi.responses import PlainTextResponse
import subprocess
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import io

import shutil
import torch
from time import  strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
import multiprocessing
import threading

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

Init = False
preprocess_model = None
audio_to_coeff = None
animate_from_coeff = None
first_coeff_path = None
crop_pic_path = None
crop_info = None
def run_inference_async(driven_audio, source_image, enhan, output):
    print("run_inference_async:",driven_audio, source_image, enhan, output)
    if torch.cuda.is_available():
        devi = "cuda"
    else:
        devi = "cpu"
    pic_path = source_image
    audio_path = driven_audio
    enhancer = enhan
    # save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    save_dir = os.path.join("./results", output)
    os.makedirs(save_dir, exist_ok=True)
    pose_style = 0
    device = devi
    batch_size = 2
    input_yaw_list = None
    input_pitch_list = None
    input_roll_list = None
    ref_eyeblink = None
    ref_pose = None
    checkpoint_dir = './checkpoints'
    size = 256
    # current_root_path = os.path.split(sys.argv[0])[0]
    current_root_path = os.path.dirname(os.path.abspath(__file__))
    old_version = False
    preprocess = 'crop'
    still = False
    expression_scale = 1
    background_enhancer = None

    ref_pose_coeff_path = None
    ref_eyeblink_coeff_path = None

    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size,
                                old_version,  preprocess)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    global Init
    global preprocess_model
    global audio_to_coeff
    global animate_from_coeff
    global first_coeff_path, crop_pic_path, crop_info
    if not Init:
        preprocess_model = CropAndExtract(sadtalker_paths, device)

        audio_to_coeff = Audio2Coeff(sadtalker_paths, device)

        animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, first_frame_dir,
                                                                                preprocess,
                                                                               source_image_flag=True,
                                                                               pic_size=size)
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return
        Init = False


    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                               expression_scale=expression_scale, still_mode=still,
                               preprocess=preprocess, size=size)

    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                         enhancer=enhancer, background_enhancer=background_enhancer,
                                         preprocess=preprocess, img_size=size)

    shutil.move(result, save_dir + '.mp4')
    print('The generated video is named:', save_dir + '.mp4')

    shutil.rmtree(save_dir)


@app.get("/process/{name}")
async def process_audio(name: str, background_tasks: BackgroundTasks):
    video_path = f"./results/{name}" + ".mp4"
    if os.path.exists(video_path):
        return PlainTextResponse(content=f"Error: file already exist!", status_code=500)

    command = f"python inference.py --driven_audio /tmp/{name}.wav --source_image demo/wangpeng.png --enhancer gfpgan --output {name}"
    print(command)

    driven_audio = f"/tmp/{name}.wav"
    source_image = "demo/wangpeng.png"
    enhancer = "gfpgan"
    output = name


    background_tasks.add_task(run_inference_async, driven_audio, source_image, enhancer, output)

    return PlainTextResponse(content="Processing started in the background.")


@app.get("/video/{video_name}")
async def download_video(video_name: str):
    video_path = f"./results/{video_name}"+".mp4"

    if os.path.exists(video_path):
        # return FileResponse(video_path, media_type="video/mp4", filename=video_name+".mp4")
        # 打开视频文件（这里假设视频文件名为 "video.mp4"）
        with open(video_path, mode="rb") as video_file:
            video_bytes = video_file.read()

        # 创建一个流式响应对象，将视频内容作为流返回给客户端
        return StreamingResponse(io.BytesIO(video_bytes), media_type="video/mp4")
    else:
        return Response(content="Video not found", status_code=404)

def run_inference_async_threaded(driven_audio, source_image, enhancer, output):
    run_inference_async(driven_audio, source_image, enhancer, output)

@app.post("/upload/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    fileMainName = file.filename.split(".")[0]

    video_path = f"./results/{fileMainName}" + ".mp4"
    if os.path.exists(video_path):
        return PlainTextResponse(content=f"Error: file already exist!", status_code=500)

    # 确保文件保存在 tmp 目录中
    upload_folder = "/tmp"
    file_path = os.path.join(upload_folder, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    command = f"python inference.py --driven_audio /tmp/{fileMainName}.wav --source_image demo/wangpeng.png --enhancer gfpgan --output {fileMainName}"
    print(command)

    driven_audio = f"/tmp/{fileMainName}.wav"
    source_image = "demo/wangpeng.png"
    enhancer = "gfpgan"
    output = fileMainName

    # background_tasks.add_task(run_inference_async, driven_audio, source_image, enhancer, output)
    # loop = asyncio.get_event_loop()
    # await loop.run_in_executor(None, run_inference_async, driven_audio, source_image, enhancer, output)
    # await run_inference_async(driven_audio, source_image, enhancer, output)

    t = threading.Thread(target=run_inference_async_threaded, args=(driven_audio, source_image, enhancer, output))
    t.start()

    return {"filename": file.filename, "file_path": file_path}