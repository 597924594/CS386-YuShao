"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'photo2sketch_train': 737,
    'photo2sketch_test': 120,
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'photo2sketch_train': '.npy',
    'photo2sketch_test': '.npy',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'photo2sketch_train': '/home/lxh/cyclegan/image_processing/input/photo2sketch/photo2sketch_train.csv',
    'photo2sketch_test': '/home/lxh/cyclegan/image_processing/input/photo2sketch/photo2sketch_test.csv',
    
}
