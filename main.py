import os
import shutil
from os.path import isfile, join

import cv2
import natsort
from moviepy.editor import *

answer = input("시작할까?")
if (answer=="0"):
    answer2 = input("지울까?")
    if (answer2=="0"):
        shutil.rmtree("./Frame")
        os.makedirs("./Frame")
    vidcap = cv2.VideoCapture('./Video/test.mp4')
    audioclip = AudioFileClip('./Video/test.mp4')
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
        hasFrames, image = vidcap.read()
        if hasFrames:
            cv2.imwrite("./Frame/" + str(count)+".png", image)
            return hasFrames
    sec = 0
    frameRate = 1/30
    count = 1
    success = getFrame(sec)
    clip = VideoFileClip("./Video/test.mp4")
    duration = float(clip.duration)*60
    # 프레임 분할
    while success:
        count = count +1
        sec = sec + frameRate
        sec = round(sec,1000)
        success = getFrame(sec)
        print(str(count) + "/" + str(duration) + "working frame")

    pathIn = "./Frame/"
    pathOut = './Result/video.mp4'
    fps = 1/frameRate

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #정렬
    files = natsort.natsorted(files)



    #비디오로 합치기
    for i in range(len(files)):
        filename = pathIn + files[i]
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'),fps,size)


    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

    videoclip = VideoFileClip('./Result/video.mp4')
    soundvideoclip = videoclip.set_audio(audioclip)
    soundvideoclip.write_videofile("./Result/soundedvideo.mp4")

