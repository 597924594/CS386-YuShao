import os
import numpy as np 
import PIL.Image

files = os.popen('dir /b /a-d')
for file in files:
    if '.jpg' in file:
        image = PIL.Image.open(file.strip())
        arr = np.array(image)
        filename = 'npyfile\\' + file.strip('.jpg\n') + '.npy'
        np.save(filename, arr)
        print(file, 'success')