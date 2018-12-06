from PIL import Image
import os
import numpy as np

filelist = os.listdir('/home/lxh/cyclegan/image_processing/rebuild/output')
for file in filelist:
	a = np.load('/home/lxh/cyclegan/image_processing/rebuild/output/'+file)
	a = (np.squeeze(a)).astype(np.uint8)
	pic = Image.fromarray(a)
	pic.save('/home/lxh/cyclegan/image_processing/rebuild/pic/'+file.strip('npy')+'jpg')