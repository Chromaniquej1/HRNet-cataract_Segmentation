import os
import numpy as np
from tensorflow.keras.utils import Sequence
from PIL import Image
import cv2
from .utils import normalize

class DataGenerator(Sequence):
    def __init__(self, img_dir, mask_dir, batch_size, num_classes, img_size, n_bands):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.img_size = img_size
        self.n_bands = n_bands
        self.file_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return int(np.ceil(len(self.file_list) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.__data_generation(batch_files)

    def __data_generation(self, batch_files):
        img = np.zeros((self.batch_size, self.img_size, self.img_size, self.n_bands), dtype='float32')
        mask = np.zeros((self.batch_size, self.img_size, self.img_size, self.num_classes), dtype='float32')

        for i, file_name in enumerate(batch_files):
            img_path = os.path.join(self.img_dir, file_name)
            image = np.array(Image.open(img_path).convert('RGB'))
            image = normalize(cv2.resize(image, (self.img_size, self.img_size)).astype('float32'))
            img[i] = image

            mask_path = os.path.join(self.mask_dir, file_name)
            mask_image = np.array(Image.open(mask_path).convert('L'))
            mask_image = cv2.resize(mask_image, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

            mask_image = np.clip(mask_image, 0, self.num_classes - 1)
            mask_one_hot = np.eye(self.num_classes)[mask_image]
            mask[i] = mask_one_hot

        return img, mask
