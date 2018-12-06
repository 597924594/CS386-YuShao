# CycleGAN in TensorFlow

**[update 9/26/2017]** We observed faster convergence and better performance after adding skip connection between input and output in the generator. To turn the feature on, use switch --skip=True. This is the result of turning on skip after training for 23 epochs:

<img src='imgs/skip_result.jpg' width="900px"/>

This is the TensorFlow implementation for CycleGAN. The code was written by [Harry Yang](https://www.harryyang.org) and [Nathan Silberman](https://github.com/nathansilberman).

CycleGAN: [[Project]](https://junyanz.github.io/CycleGAN/) [[Paper]](https://arxiv.org/pdf/1703.10593.pdf) 

## Introduction

This code contains two versions of the network architectures and hyper-parameters. The first one is based on the [TensorFlow implementation](https://github.com/hardikbansal/CycleGAN). The second one is based on the [official PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The differences are minor and we observed both versions produced good results. You may need to train several times as the quality of the results are sensitive to the initialization.  

Below is a snapshot of our result at the 50th epoch on one training instance:

<img src='imgs/horse2zebra.png' width="900px"/>

## Getting Started
### Prepare dataset
* You can either download one of the defaults CycleGAN datasets or use your own dataset. 
	* Download a CycleGAN dataset (e.g. horse2zebra):
	```bash
	bash ./download_datasets.sh horse2zebra
	```
	* Use your own dataset: put images from each domain at folder_a and folder_b respectively. 

* Create the csv file as input to the data loader. 
	* Edit the cyclegan_datasets.py file. For example, if you have a face2ramen_train dataset which contains 800 face images and 1000 ramen images both in PNG format, you can just edit the cyclegan_datasets.py as following:
	```python
	DATASET_TO_SIZES = {
    'face2ramen_train': 1000
	}

	PATH_TO_CSV = {
    'face2ramen_train': './CycleGAN/input/face2ramen/face2ramen_train.csv'
	}

	DATASET_TO_IMAGETYPE = {
    'face2ramen_train': '.png'
	}

	``` 
	* Run create_cyclegan_dataset.py:
	```bash
	python -m 3dnew.create_cyclegan_dataset --image_path_a=/disk/lym/cyclegan/cyclegan/lkds0724 --image_path_b=/disk/lym/cyclegan/cyclegan/shxk0724 --dataset_name="lkds2shxk_train" --do_shuffle=0
	```
	python -m CycleGAN_TensorFlow3d.create_cyclegan_dataset --image_path_a=/disk/lym/cyclegan/cyclegan/lkds_up_test --image_path_b=/disk/lym/cyclegan/cyclegan/shxk_up_test --dataset_name="lkds2shxk_test" --do_shuffle=0

### Training
* Create the configuration file. The configuration file contains basic information for training/testing. An example of the configuration file could be fond at configs/exp_01.json. 

* Start training:
```bash
python2 -m CycleGAN_TensorFlow_new.main\
    --to_train=1 \
    --log_dir=CycleGAN_TensorFlow_new/output/cyclegan/exp_05 \
    --config_filename=CycleGAN_TensorFlow_new/configs/exp_05.json
```
python2 -m CycleGAN_WMD.main --to_train=1 --log_dir=CycleGAN_WMD/output/cyclegan/exp_05 --config_filename=CycleGAN_WMD/configs/exp_05.json

* Check the intermediate results.
	* Tensorboard
	```bash
	tensorboard --port=6006 --logdir=CycleGAN_TensorFlow/output/cyclegan/exp_5/#timestamp# 
	```
	* Check the html visualization at CycleGAN_TensorFlow/output/cyclegan/exp_01/#timestamp#/epoch_#id#.html.  

### Restoring from the previous checkpoint.
```bash
python -m CycleGAN_TensorFlow.main \
    --to_train=2 \
    --log_dir=CycleGAN_TensorFlow/output/cyclegan/exp_05 \
    --config_filename=CycleGAN_TensorFlow/configs/exp_05.json \
    --checkpoint_dir=CycleGAN_TensorFlow/output/cyclegan/exp_01/20180516-135806
```
### Testing
* Create the testing dataset.
	* Edit the cyclegan_datasets.py file the same way as training.
	* Create the csv file as the input to the data loader. 
	```bash
	python -m CycleGAN_TensorFlow_new.create_cyclegan_dataset --image_path_a=/home/lxh/cyclegan/lkds_up_test_0703 --image_path_b=/home/lxh/cyclegan/shxk_up_test_0703--dataset_name="lkds2shxk_test" --do_shuffle=0
	```
* Run testing.
```bash
CUDA_VISIBLE_DEVICES=0 python2 -m CycleGAN_TensorFlow_new.main \
    --to_train=0 \
    --log_dir=CycleGAN_TensorFlow_WMD/output/cyclegan/exp_05 \
    --config_filename=CycleGAN_TensorFlow_WMD/configs/exp_05_test.json \
    --checkpoint_dir=CycleGAN_TensorFlow_WMD/output/cyclegan/exp_05/20180705-160808 

### 看这里看这里

CUDA_VISIBLE_DEVICES=3 python2 -m image_processing.main --to_train=1 --log_dir=image_processing/output/cyclegan/exp_05 --config_filename=image_processing/configs/exp_05.json --checkpoint_dir=image_processing/output/cyclegan/exp_05/20181006-151308

    nvidia-smi 反
```
### HERE！！HERE！！

The result is saved in CycleGAN_TensorFlow/output/cyclegan/exp_01/#new_timestamp#.




/disk/lym/cyclegan/cyclegan/CycleGAN_TensorFlow3d/output/cyclegan/exp_05/20180612-102359
