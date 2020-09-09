# resize pokeGAN.py
import os
import cv2
from os.path import isdir

src = "./data" #pokeRGB_black
dst = "./resizedData" # resized

os.mkdir(dst)

i=0
for each in os.listdir(src):
    i+=1
    print(str(i)+ " worked")
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(dst,each), img)
    