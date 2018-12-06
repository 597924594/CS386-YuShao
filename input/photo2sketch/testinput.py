f = open('photo2sketch_train.csv')
lines = f.readlines()
for line in lines:
	tmp = line.strip().split(',')
	if '.npy' in tmp[1]:
		whether = True
	else:
		whether = False
	print(tmp, whether)
