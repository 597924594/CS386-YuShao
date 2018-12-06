import numpy as np 
from PIL import Image as image

def sharper(img, x, y, channel):
    def value(img, x, y, channel):
        if x<=0 or x>=255 or y<=0 or y >= 255:
            return 0
        else:
            return img[x][y][channel]
    return 5*value(img, x, y, channel) - value(img, x-1, y, channel) - \
            value(img, x+1, y, channel) - value(img, x, y+1, channel) - \
            value(img, x, y-1, channel)

a=np.squeeze(np.load('output/1_fake_sketch.npy'))
print(a.shape)
ans=np.zeros((256,256,3))
for x in range(256):
    for y in range(256):
        for channel in range(3):
            ans[x][y][channel]=sharper(a, x, y, channel)
ans[ans<0]=0
ans[ans>255]=255
ans = ans.astype(np.uint8)
pic = image.fromarray(a.astype(np.uint8))
pic.save('test_former.jpg')

pic = image.fromarray(ans)
pic.save('test_sharped.jpg')

b=np.squeeze(np.load('output/1_input_sketch.npy'))
pic = image.fromarray(b)
pic.save('test_input.jpg')