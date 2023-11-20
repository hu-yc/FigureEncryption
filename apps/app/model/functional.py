import os
import cv2
import math
import time
import pathlib
import hashlib
import numpy as np
from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt

from model.resnet import get_resnet
from utils.utils_log import LogFactory


log = LogFactory.get_log('audit')


class CallResNet:
    def __init__(self, weight_path='./conf/d_A_epoch100.h5'):
        self.model = get_resnet()
        self.encrypted_figure = np.zeros((256, 256, 3), dtype=np.int32)
        self.raw_seed = np.random.randint(0, 20000001, size=(1, 256, 256, 3))
        self.model.load_weights(weight_path, skip_mismatch=True, by_name=True)

    def initialized_seed(self):
        log.info("[initialized_seed] start to initialize the random seed")
        random_seed = np.array(self.raw_seed, dtype=np.float32) / 10000000 - 1
        log.info("[initialized_seed] generate the random seed successfully")
        return random_seed

    def random_image_generator(self):
        random_seed = self.initialized_seed()
        log.info("[random_image_generator] start to generate the random image")
        latent_fake = np.int32(self.model.predict(random_seed))
        log.info("[random_image_generator] generate the random image successfully")
        return latent_fake


class ImageInteract:
    def __init__(self, img_path, o_path="/outputs"):
        self.img_hash = None
        self.raw_img_size = None
        self.img_path = img_path
        self.output_path = o_path
        self.encrypted_figure = None
        self.decrypted_figure = None
        self.entropy_matrix = np.zeros(256, dtype=np.float32)
        self.img_filetype = pathlib.Path(self.img_path).suffix

    def cal_hash(self):
        log.info("[cal_hash] start to calculate the md5 of input image")
        with open(self.img_path, 'rb') as f:
            md5 = hashlib.md5(f.read())
        md5_hash = md5.hexdigest()
        log.info("[cal_hash] the md5 of input image is {}".format(md5_hash))
        return md5_hash

    def cvt_img(self):
        log.info("[cvt_img] start to convert the input image from BGR to RGB")
        raw_img = cv2.imread(self.img_path)
        self.img_hash = self.cal_hash()
        self.raw_img_size = raw_img.shape
        log.info("[cvt_img] get the raw image size: {}".format(self.raw_img_size))
        cvt_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        log.info("[cvt_img] convert the input image to RGB")
        return cvt_img

    @staticmethod
    def element_xor(first_array, second_array):
        array_size = first_array.size
        output_array = np.zeros(array_size, dtype=np.int32)
        for e in range(array_size):
            output_array[e] = first_array[e] ^ second_array[e]
        return output_array

    def sliding_window_differential(self, raw_array, random_array, raw_size, random_size, flag):
        indexes_list = []
        times = range(math.floor(raw_size / random_size))
        log.debug("[sliding_window_differential] the number of times is {}".format(len(times)))
        num_remain_elements = raw_size % random_size
        log.debug("[sliding_window_differential] the number of remain elements is {}".format(num_remain_elements))
        for j in times:
            start_index = j * random_size + num_remain_elements
            end_index = start_index + random_size
            indexes_list.append([start_index, end_index])
        log.debug("[sliding_window_differential] the list of index: {}".format(indexes_list))
        iterative_index = [[0, random_size]] + indexes_list
        if not flag:
            iterative_index.reverse()
        log.debug("[sliding_window_differential] the list of iterative index: {}".format(iterative_index))
        for start_index, end_index in iterative_index:
            raw_array[start_index:end_index] = self.element_xor(raw_array[start_index:end_index],
                                                                random_array)
        return raw_array

    def differential_fig(self, raw_img, random_img, encrypt_flag):
        o_h, o_w, o_c = raw_img.shape
        mid_img = random_img[0, :, :, :]
        output_figure = np.zeros((o_h, o_w, o_c), dtype=np.int32)
        for i in range(o_c):
            e_raw_img = raw_img[:, :, i].flatten()
            e_random_img = mid_img[:, :, i].flatten()
            origin_size = e_raw_img.size
            random_size = e_random_img.size
            if origin_size > random_size:
                e_raw_img = self.sliding_window_differential(e_raw_img, e_random_img, origin_size, random_size,
                                                             encrypt_flag)
            else:
                e_random_img = e_random_img[:origin_size]
                e_raw_img = self.element_xor(e_raw_img, e_random_img)
            output_figure[:, :, i] = e_raw_img.reshape(o_h, o_w)
        return output_figure

    def encrypt_fig(self, input_random_img):
        log.info("[encrypt_fig] start to encrypt the input image")
        origin_img = self.cvt_img()
        start_time = time.time()
        self.encrypted_figure = self.differential_fig(origin_img, input_random_img, 1)
        end_time = time.time()
        encrypt_period = end_time - start_time
        log.info("[encrypt_fig] the period of encrypting the input image is {}".format(encrypt_period))
        encrypted_fig = Image.fromarray(np.uint8(self.encrypted_figure))
        encrypted_fig_path = os.path.join(self.output_path,
                                          "encrypted_{}_{}.{}".format(self.img_hash,
                                                                      datetime.now().strftime("%Y%m%d%H%M%S"),
                                                                      self.img_filetype))
        encrypted_fig.save(encrypted_fig_path)
        log.info("[encrypt_fig] save the encrypted image to {}".format(encrypted_fig_path))
        return encrypt_period, encrypted_fig_path

    def decrypt_fig(self, input_random_img):
        log.info("[decrypt_fig] start to decrypt the encrypted image")
        start_time = time.time()
        self.decrypted_figure = self.differential_fig(self.encrypted_figure, input_random_img, 0)
        end_time = time.time()
        decrypt_period = end_time - start_time
        log.info("[decrypt_fig] the period of decrypting the encrypted image is {}".format(decrypt_period))
        decrypted_fig = Image.fromarray(np.uint8(self.decrypted_figure))
        decrypted_fig_path = os.path.join(self.output_path,
                                          "decrypted_{}_{}.{}".format(self.img_hash,
                                                                      datetime.now().strftime("%Y%m%d%H%M%S"),
                                                                      self.img_filetype))
        decrypted_fig.save(decrypted_fig_path)
        log.info("[decrypt_fig] save the decrypted image to {}".format(decrypted_fig_path))
        return decrypt_period, decrypted_fig_path

    def cal_entropy(self):
        entropy_value = 0
        log.info("[cal_entropy] start to calculate the entropy of encrypted image")
        e_h, e_w, e_c = self.encrypted_figure.shape
        encrypt_fig_size = e_h * e_w * e_c
        for c in range(e_c):
            e_encrypt_img = np.int32(self.encrypted_figure[:, :, c].flatten())
            for e_value in e_encrypt_img:
                self.entropy_matrix[e_value] += 1
        for value_index in range(256):
            if self.entropy_matrix[value_index] != 0:
                entropy_value += - (self.entropy_matrix[value_index] / encrypt_fig_size) * (
                        math.log(self.entropy_matrix[value_index] / encrypt_fig_size) / math.log(2.0))
        return entropy_value

    def plot_performance_indicator(self, encrypt_period, decrypt_period, entropy_value):
        log.info("[plot_performance_indicator] start to plot the performance indicator")
        plt.figure()
        plt.annotate('encryption_time: {}, \ndecryption_time: {}, \ninformation_entropy: {}'.format(
            round(encrypt_period, 4), round(decrypt_period, 4), round(entropy_value, 4)),
            xy=(0, 0), xytext=(0, 1.2), textcoords="axes fraction", fontsize=12, xycoords="axes fraction",
            ha="left", va="bottom")
        log.info("[estimate_performance] adding the annotation: {}, {}, {}".format(round(encrypt_period, 4),
                                                                                   round(decrypt_period, 4),
                                                                                   round(entropy_value, 4)))
        plt.title("Grayscale Histogram of Ciphertext Image")
        plt.xlabel("Pixel Value")
        plt.ylabel("Numbers of Pixels")
        plt.bar(range(256), self.entropy_matrix, width=1)
        plt.xlim([0, 255])
        plt.ylim([0, 1600])
        performance_fig_path = os.path.join(self.output_path,
                                            "performance_{}_{}.png".format(self.img_hash,
                                                                           datetime.now().strftime("%Y%m%d%H%M%S")))
        plt.savefig(performance_fig_path, bbox_inches='tight', pad_inches=0)
        log.info("[estimate_performance] save the performance figure to {}".format(performance_fig_path))
        return performance_fig_path





