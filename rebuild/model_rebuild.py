import tensorflow as tf
import numpy as np
import os
from PIL import Image

# with densenet
MODEL_PATH = '/home/lxh/cyclegan/image_processing/rebuild/cyclegan-27'
SAVE_PATH = '/home/lxh/cyclegan/image_processing/rebuild'
PHOTO_PATH = '/home/lxh/cyclegan/image_processing/data/train/photos/npyfile'
SKETCH_PATH = '/home/lxh/cyclegan/image_processing/data/train/sketches/npyfile'
photos = os.listdir(PHOTO_PATH)
sketches = os.listdir(SKETCH_PATH)

saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
# g_A = rebuild_generator_tf(input_tensor, name="g_A")
# g_B = rebuild_generator_tf(input_tensor, name="g_B")


if not os.path.exists(os.path.join(SAVE_PATH, 'output')):
    os.makedirs(os.path.join(SAVE_PATH, 'output'))
# if not os.path.exists(os.path.join(SAVE_PATH, 'fake_sketch')):
#     os.makedirs(os.path.join(SAVE_PATH, 'fake_sketch'))

print 'sess begin'

with tf.Session() as sess:
    saver.restore(sess, MODEL_PATH)
    print 'module loaded'
    # input_A = tf.get_default_graph().get_tensor_by_name("input_A:0")
    # input_B = tf.get_default_graph().get_tensor_by_name("input_B:0")
    input_A = tf.get_default_graph().get_tensor_by_name("g_A_input:0")
    input_B = tf.get_default_graph().get_tensor_by_name("g_B_input:0")

    for name in sketches:
        img = np.load(os.path.join(SKETCH_PATH, name))
        img = (img/255.0)
        img = img.reshape((1, 256, 256, 3, 1))
        # print (img.shape, img.dtype)
        result = sess.run(tf.get_default_graph().get_tensor_by_name('Model/g_B/c24/relu:0'),
            feed_dict={
                input_B: img
            })
        np.save(os.path.join(SAVE_PATH, 'output', name.strip('.npy')+'_fake_photo.npy'), (result*255.0).astype(np.int32))
        print name+' finished'
        np.save(os.path.join(SAVE_PATH, 'output', name.strip('.npy')+'_input_sketch.npy'), (img*255.0).astype(np.uint8))

    for name in photos:
        img = np.load(os.path.join(PHOTO_PATH, name))
        img = (img/255.0)
        img = img.reshape((1, 256, 256, 3, 1))
        result = sess.run(tf.get_default_graph().get_tensor_by_name('Model/g_A/c24/relu:0'),
            feed_dict={
                input_A: img
            })
        np.save(os.path.join(SAVE_PATH, 'output', name.strip('.npy')+'_fake_sketch.npy'), (result*255.0).astype(np.int32))
        print name+' finished'
        np.save(os.path.join(SAVE_PATH, 'output', name.strip('.npy')+'_input_photo.npy'), (img*255.0).astype(np.uint8))


filelist = os.listdir('/home/lxh/cyclegan/image_processing/rebuild/output')
for file in filelist:
    a = np.load('/home/lxh/cyclegan/image_processing/rebuild/output/'+file)
    a = (np.squeeze(a))
    a[a<0]=0
    a[a>255]=255
    a=a.astype(np.uint8)
    pic = Image.fromarray(a)
    pic.save('/home/lxh/cyclegan/image_processing/rebuild/pic/'+file.strip('npy')+'jpg')