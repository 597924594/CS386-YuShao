import tensorflow as tf
import numpy as np
import pdb
from . import cyclegan_datasets
from . import model

half_input = {}


def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)

    if image_type == '.jpg':
        image_decoded_A = tf.image.decode_jpeg(
            file_contents_i, channels=model.IMG_CHANNELS)
        image_decoded_B = tf.image.decode_jpeg(
            file_contents_j, channels=model.IMG_CHANNELS)
    elif image_type == '.npz':
        image_decoded_A = np.load(file_contents_i)
        image_decoded_B = np.load(file_contents_j)
    return image_decoded_A, image_decoded_B


def load_data(dataset_name, image_size_before_crop,
              do_shuffle=True, do_flipping=False):
    """

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    image_i, image_j = _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    inputs = {
        'image_i': image_i,
        'image_j': image_j
    }

    # Preprocessing:
    inputs['image_i'] = tf.image.resize_images(
        inputs['image_i'], [image_size_before_crop, image_size_before_crop])
    inputs['image_j'] = tf.image.resize_images(
        inputs['image_j'], [image_size_before_crop, image_size_before_crop])

    if do_flipping is True:
        inputs['image_i'] = tf.image.random_flip_left_right(inputs['image_i'])
        inputs['image_j'] = tf.image.random_flip_left_right(inputs['image_j'])

    inputs['image_i'] = tf.random_crop(
        inputs['image_i'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    inputs['image_j'] = tf.random_crop(
        inputs['image_j'], [model.IMG_HEIGHT, model.IMG_WIDTH, 3])

    inputs['image_i'] = tf.subtract(tf.div(inputs['image_i'], 127.5), 1)
    inputs['image_j'] = tf.subtract(tf.div(inputs['image_j'], 127.5), 1)

    # Batch
    if do_shuffle is True:
        inputs['images_i'], inputs['images_j'] = tf.train.shuffle_batch(
            [inputs['image_i'], inputs['image_j']], 1, 5000, 100)
    else:
        inputs['images_i'], inputs['images_j'] = tf.train.batch(
            [inputs['image_i'], inputs['image_j']], 1)

    return inputs
# not used above


class data_loader:
    def __init__(self, dataset_name, do_shuffle):
        csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]
        print(csv_name)
        self.filesPath = csv_name
        self.listI = []
        self.listJ = []
        self.do_shuffle = do_shuffle
        self.getFilePath()

    def getFilePath(self):
        with open(self.filesPath, 'r') as f:
            lines = f.readlines()
            if self.do_shuffle:
                np.random.shuffle(lines)

            for line in lines:
                arr = line.strip().split(',')
                if '.npy' not in arr[1]:
                    continue
                self.listI.append(arr[0])
                self.listJ.append(arr[1])
        print len(self.listI)
        print len(self.listJ)

    def getBatch(self, i):
        # pdb.set_trace()
        print (self.listI[i], self.listJ[i])
        print '***********'
        modulei = np.load(self.listI[i])
        modulej = np.load(self.listJ[i])
        inputs = {
            'photo': np.reshape(modulei/255.0, (1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, 1)),
            'sketch': np.reshape(modulej/255.0, (1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_DEPTH, 1)),
        }
        # imi=np.swapaxes((np.load(self.listI[i])/255.0),0,2)
        # imj=np.swapaxes((np.load(self.listJ[i])/255.0),0,2)
        # inputs = {
        # 'images_i': np.reshape(imi,(1,48,48,48,1)),
        # 'images_j': np.reshape(imj,(1,48,48,48,1)),
        # }

        return inputs
