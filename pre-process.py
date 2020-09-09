from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import natsort
import os
from os.path import isfile, join

# 랜덤시드 고정시키기
np.random.seed(5)

# 데이터셋 불러오기
data_aug_gen = ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=15,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8, 2.0],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')

path = "./png_dataset/"
files = [f for f in os.listdir(path) if isfile(join(path, f))]
files = natsort.natsorted(files)
for j in range(len(files)):

    img = load_img(path + files[j])
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0

    # 이 for는 무한으로 반복되기 때문에 우리가 원하는 반복횟수를 지정하여, 지정된 반복횟수가 되면 빠져나오도록 해야합니다.
    for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir='./png_data_resized', save_prefix='shio_p_dset', save_format='png'):
        i += 1
        print(str(i) +" gen")
        if i > 15:
            break