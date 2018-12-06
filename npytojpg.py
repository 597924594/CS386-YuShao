##########
#        #
#for show_data#
#        #
##########
# import numpy as np
# from PIL import Image

# lkds='/home/lxh/cyclegan/lkds_up_test_0703/'

# for j in range (1,600):
#   # if (j==703 or j==652 or j==411 or j==430 or j==490 or j==492 or j==495 or j==507  or j==510 or j==514 or j==567 or j==603):
#       # continue
#   nodule=np.load(lkds+str(j)+'.npy')
#   for i in range (len(nodule)):
#       npy = Image.fromarray(nodule[i])
#       npy.save('/home/lxh/cyclegan/show_data/'+str(j)+'_'+str(i)+'.jpg')

##########
#        #
#for train#
#        # 	
##########

import numpy as np
from PIL import Image
import os
num = 27
imgnum = 0
time = '20181119-182753'
file_path = '/home/lxh/cyclegan/image_processing/output/cyclegan/exp_05/'+time+'/imgs/'
pic_path = '/home/lxh/cyclegan/image_processing/check_train/' + time +'/'

print pic_path

if not os.path.exists(pic_path):
    os.makedirs(pic_path)

inputB = np.squeeze(np.load(file_path+'inputB_'+str(num)+'_'+str(imgnum)+'.npy'))
inputA = np.squeeze(np.load(file_path+'inputA_'+str(num)+'_'+str(imgnum)+'.npy'))
fakeA = np.squeeze(np.load(file_path+'fakeA_'+str(num)+'_'+str(imgnum)+'.npy'))
fakeB = np.squeeze(np.load(file_path+'fakeB_'+str(num)+'_'+str(imgnum)+'.npy'))
cycA = np.squeeze(np.load(file_path+'cycA_'+str(num)+'_'+str(imgnum)+'.npy'))
cycB = np.squeeze(np.load(file_path+'cycB_'+str(num)+'_'+str(imgnum)+'.npy'))
inputA[inputA > 255] = 255
a = 0
# for i in range(len(inputB)):
#     npy = Image.fromarray(inputB[i])
#     a += 1
#     npy.save(pic_path+'inputb'+'_'+str(a)+'.jpg')
img = Image.fromarray(inputB)
img.save(pic_path+'inputb_all.jpg')

# b = 0
# for i in range(len(inputA)):
#     npy = Image.fromarray(inputA[i])
#     b += 1
#     npy.save(pic_path + 'inputa' + '_' + str(b) + '.jpg')
img = Image.fromarray(inputA)
img.save(pic_path+'inputa_all.jpg')

# c = 0
# for i in range(len(fakeA)):
#     npy = Image.fromarray(fakeA[i])
#     c += 1
#     npy.save(pic_path + 'fakea' + '_' + str(c) + '.jpg')
img = Image.fromarray(fakeA)
img.save(pic_path+'fakea_all.jpg')

# d = 0
# for i in range(len(fakeB)):
#     npy = Image.fromarray(fakeB[i])
#     d += 1
#     npy.save(pic_path + 'fakeb' + '_' + str(d) + '.jpg')
img = Image.fromarray(fakeB)
img.save(pic_path+'fakeb_all.jpg')

# e = 0
# for i in range(len(cycA)):
#     npy = Image.fromarray(cycA[i])
#     e += 1
#     npy.save(pic_path + 'cyca' + '_' + str(e) + ' .jpg')
img = Image.fromarray(cycA)
img.save(pic_path+'cyca_all.jpg')

# f = 0
# for i in range(len(cycB)):
#     npy = Image.fromarray(cycB[i])
#     f += 1
#     npy.save(pic_path + 'cycb' + '_' + str(f) + ' .jpg')
img = Image.fromarray(cycB)
img.save(pic_path+'cycb_all.jpg')
print 'end of npytojpg'

##########
#        #
#for test#
#        #
##########

# import numpy as np
# from PIL import Image
# num=0
# imgnum=0
# file_path='/home/lxh/cyclegan/CycleGAN_TensorFlow_new/output/cyclegan/exp_05/20180709-142249/imgs/'
# for num in range(0,99):
#   inputB=np.squeeze(np.load(file_path+'inputB_'+str(num)+'_'+str(imgnum)+'.npy'))
#   inputA=np.squeeze(np.load(file_path+'inputA_'+str(num)+'_'+str(imgnum)+'.npy'))
#   fakeA=np.squeeze(np.load(file_path+'fakeA_'+str(num)+'_'+str(imgnum)+'.npy'))
#   fakeB=np.squeeze(np.load(file_path+'fakeB_'+str(num)+'_'+str(imgnum)+'.npy'))
#   cycA=np.squeeze(np.load(file_path+'cycA_'+str(num)+'_'+str(imgnum)+'.npy'))
#   cycB=np.squeeze(np.load(file_path+'cycB_'+str(num)+'_'+str(imgnum)+'.npy'))

#   a=0
#   for i in range (len(inputB)):
#       npy = Image.fromarray(inputB[i])
#       a+=1
#       npy.save('/home/lxh/cyclegan/check_test/inputb'+str(num)+'_'+str(a)+'.jpg')
#   b=0
#   for i in range (len(inputA)):
#       npy = Image.fromarray(inputA[i])
#       b+=1
#       npy.save('/home/lxh/cyclegan/check_test/inputa'+str(num)+'_'+str(b)+'.jpg')
#   c=0
#   for i in range (len(fakeA)):
#       npy = Image.fromarray(fakeA[i])
#       c+=1
#       npy.save('/home/lxh/cyclegan/check_test/fakea'+str(num)+'_'+str(c)+'.jpg')
#   d=0
#   for i in range (len(fakeB)):
#       npy = Image.fromarray(fakeB[i])
#       d+=1
#       npy.save('/home/lxh/cyclegan/check_test/fakeb'+str(num)+'_'+str(d)+'.jpg')
#   e=0
#   for i in range (len(cycA)):
#       npy = Image.fromarray(cycA[i])
#       e+=1
#       npy.save('/home/lxh/cyclegan/check_test/cyca'+str(num)+'_'+str(e)+'.jpg')
#   f=0
#   for i in range (len(cycB)):
#       npy = Image.fromarray(cycB[i])
#       f+=1
#       npy.save('/home/lxh/cyclegan/check_test/cycb'+str(num)+'_'+str(f)+'.jpg')


