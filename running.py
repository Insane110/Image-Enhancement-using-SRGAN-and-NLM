
#importing libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

#mounting drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.models import model_from_json
# load json and create model
json_file = open('/content/model_34_.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/model_34_.h5")

#created function for giving input and output images
def convert(fpath):
    for fname in os.listdir(fpath):
        path = os.path.join(fpath,fname)
        img = cv2.imread(path)

        lr_img = cv2.resize(img,(32,32))
        hr_img = cv2.resize(img,(128,128))
        lr_img = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)

        lr_img = lr_img / 255.
        hr_img = hr_img / 255.
        lr_img = np.expand_dims(lr_img,axis=0)
        hr_img = np.expand_dims(lr_img,axis=0)

        gen_img = loaded_model.predict(lr_img)
        sigma_est = np.mean(estimate_sigma(gen_img[0,:,:,:], multichannel=True))
        patch_kw = dict(patch_size=5,      # 5x5 patches
                        patch_distance=6,  # 13x13 search area
                        channel_axis=-1)
        # fast algorithm
        denoise_fast = denoise_nl_means(gen_img, h=0.8 * sigma_est, fast_mode=True,
                                        **patch_kw)
        
        lr_img = cv2.convertScaleAbs(lr_img, alpha=(255.0))
        denoise_fast = cv2.convertScaleAbs(denoise_fast, alpha=(255.0))
        # fast algorithm, sigma provided
        # denoise2_fast = denoise_nl_means(gen_img, h=0.8 * sigma_est, sigma=sigma_est,
        #                                 fast_mode=True, **patch_kw)
        # show(gen_img,lr_img,hr_img)
        dataset_name1 = "saved images"
        dataset_name2 = "low res images"
        out_dir1 = os.path.join("/content/output/",dataset_name1 )
        out_dir2 = os.path.join("/content/input/",dataset_name2 )
        if not os.path.exists(out_dir1): 
            os.makedirs(out_dir1)
        if not os.path.exists(out_dir2):
            os.makedirs(out_dir2)
        cv2.imwrite(os.path.join(out_dir1, fname),denoise_fast)
        cv2.imwrite(os.path.join(out_dir2, fname),lr_img[0,:,:,:])

#running the function
convert('/content/testing')

