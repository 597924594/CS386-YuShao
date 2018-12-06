import numpy as np
import cv2
import scipy.ndimage
import growing
import pywt
lkds_mask=np.load('./lkds_nodules_answer.npy')
shxk_mask=np.load('./shxk_nodules_answer.npy')
lkds_nodule=np.load('./lkds_nodules_non182.npy')
shxk_nodule=np.load('./shxk_nodules_non182.npy')

for i in range (700,802):
	nodule_48=shxk_nodule[i][40:88,40:88,40:88]
	mask_48=shxk_mask[i][40:88,40:88,40:88]
	axis_x=[]
	axis_y=[]
	axis_z=[]
	print i
	if (i==703 or i==758 or i==411 or i==430 or i==490 or i==492 or i==495 or i==507  or i==510 or i==514 or i==567 or i==603):
		continue
	for s in range (48):
		for m in range(48):
			for k in range(48):
				if (mask_48[s][m][k]!=0):
					# print i,s,m,k
					mask_48[s][m][k]=1
					axis_x.append(s)
					axis_y.append(m)
					axis_z.append(k)
	max_tmp=[np.max(axis_x)-24,24-np.min(axis_x),np.max(axis_y)-24,24-np.min(axis_y),np.max(axis_y)-24,24-np.min(axis_y)]
	# x=np.max(axis_x)-np.min(axis_x)
	# y=np.max(axis_y)-np.min(axis_y)
	# z=np.max(axis_z)-np.min(axis_z)
	tmp=np.max(max_tmp)
	# if (x>=y):
	# 	max_temp=x
	# else:
	# 	max_temp=y
	# if (z>max_temp):
	# 	max_temp=z
	# if (max_temp%2==1):
		# max_temp+=1
	# tmp=max_temp/2
	onlynodule=nodule_48[22-tmp:26+tmp,22-tmp:26+tmp,22-tmp:26+tmp]
	mask=mask_48[22-tmp:26+tmp,22-tmp:26+tmp,22-tmp:26+tmp]
	# onlynodule=nodule*mask
	multiple=48./len(mask)
	output=scipy.ndimage.zoom(onlynodule,multiple,order=1)
	# loutput=np.concatenate((lonlynodule,lnodule))
	# np.save('../lkds_train96/'+str(i)+'.npy',loutput)
	# snodule=shxk_nodule[i][40:88,40:88,40:88]
	# smask=shxk_mask[i][40:88,40:88,40:88]
	# sonlynodule=snodule*smask
	# for j in range (len(onlynodule)):
		# up1=cv2.pyrUp(lonlynodule[j])
		# # nodule[i]=up1[24:72,24:72]
		# up2=cv2.pyrUp(up1)
		# lonlynodule[j]=up2[72:120,72:120]
		# output=cv2.resize(onlynodule,(48,48,48))
	# soutput=np.concatenate((sonlynodule,snodule))
	
	np.save('./shxk_up_test_0703/'+str(i-700)+'.npy',output)