import os
import cv2
import math
import torch
import random
import numbers
import numpy as np
from PIL import Image
from numpy.random import randint
from skimage import io, transform, color
from scipy.ndimage import morphology
 

import torch.utils.data as data


def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w,h), interpolation=cv2.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x


def maybe_random_interp(cv2_interp):
    return cv2_interp


# import imgaug
# class DataAudment(object):

#     def __init__(self, degrees,):
#         self.degrees = degrees
#     def __call__(self, sample):
#         input_t, target_t = sample['image'], sample['label']
        
#         def gamma_map(image, gamma):
#             image = np.float32(image) / 255.0
#             if gamma == 2:
#                 image = np.multiply(image, np.square(image))
#             image = np.sqrt(image + 1e-14)
#             return image * 255.0

#         if augment:
#             gamma = np.random.randint(0, 3)
#             if gamma >= 2:
#                 input_t = gamma_map(input_t, gamma)

#             aug_sequence = imgaug.augmenters.SomeOf((1, None), [ # 每次选择一个功能
#                 # imgaug.augmenters.Fliplr(0.5),
#                 imgaug.augmenters.Affine(rotate=(-30.0, 30.0))
#             ], random_order=False)
#             aug_sequence = aug_sequence.to_deterministic() # apply same transformations to mask and images

#             contrast_aug = imgaug.augmenters.ContrastNormalization((0.5, 1.5), per_channel=0.5)
#             input_t = contrast_aug.augment_images([input_t])[0]

#             image = aug_sequence.augment_images([image])[0]
#             mask = aug_sequence.augment_images([mask])[0]

        
#         sample['fg'], sample['alpha'] = fg, alpha

#         return sample

class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        # input_t [image3, trimap1]
        # target [alpha1, mask1, fg3, bg3, image3]
        input_t, target_t = sample['image'], sample['label']
        # if alpha is all 0 skip
        # al_gt = target_t[:, :, 0]
        al_gt = target_t
        fg = input_t[:, :, :3]
        if np.all(al_gt==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][al_gt > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][al_gt > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        input_t[:, :, :3] = fg*255
        sample['image'], sample['laebl'] = input_t, target_t 

        return sample



# RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None: # rand(2)表示产生两个(0,1)之间的值
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1 # 归一化到[-1, 1], -1和1排列组合，

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        input_t, target_t = sample['image'], sample['label']
        h, w, c = input_t.shape

        if np.random.rand(1) > 0.5:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, input_t.size)
        else:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, input_t.size)

        # if np.maximum(h, w) < 1024:
        #     params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        # else:
        #     params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (w * 0.5 + 0.5, h * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        input_t = cv2.warpAffine(input_t, M, (w, h),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        target_t = cv2.warpAffine(target_t, M, (w, h),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['image'], sample['label'] = input_t, target_t
        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size # 320

    def __call__(self, sample):
        input_t, target_t = sample['image'], sample['label']

        h, w = input_t.shape[:2]

        if h > w:
            new_h, new_w = self.output_size*h/w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size*w/h

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        input_t = cv2.resize(input_t, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        target_t = cv2.resize(target_t, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return {'image':input_t,'label':target_t}

class RandomCrop(object):

    def __init__(self, crop_size=320, scales=[0.5, 0.75, 1, 1.25, 1.5, 2], multicrop=False):
        self.crop_size = crop_size
        self.scales = scales
        self.multicrop = multicrop
    def __call__(self,sample):
        input_t, target_t = sample['image'], sample['label']
        crop_s = self.crop_size
        target_t = np.expand_dims(target_t, -1)
        
        if random.random() >= 0.5:
            input_t = input_t[::-1]
            target_t = target_t[::-1]

        h, w = input_t.shape[:2]
        if min(h, w) < crop_s:
            s = (crop_s + int(crop_s * 1.2)) / min(h, w)
            h = int(h*s)
            w = int(w*s)
            input_t = cv2.resize(input_t, (w, h), interpolation=cv2.INTER_CUBIC)
            target_t = cv2.resize(target_t, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # multi_size crop
        if self.multicrop:
            crop_ss = np.floor(crop_s * np.array(self.scales)).astype('int')
            crop_ss = crop_ss[crop_ss < min(h, w)]
            crop_s = int(random.choice(crop_ss))

        h1 = np.random.randint(0, h - crop_s)
        w1 = np.random.randint(0, w - crop_s)

        input_t = input_t[h1: h1 + crop_s, w1: w1 + crop_s, :]
        target_t = target_t[h1: h1 + crop_s, w1: w1 + crop_s, :]

        if crop_s != self.crop_size:
            nh = nw = self.crop_size
            input_t = cv2.resize(input_t, (nw, nh), interpolation=cv2.INTER_CUBIC)
            target_t = cv2.resize(target_t, (nw, nh), interpolation=cv2.INTER_CUBIC)

        return {'image':input_t, 'label':target_t}

class MultiScaleCrop(object):
    ''' 
        由于crop的比例必须覆盖90%, 所以必须resize和crop同时操作，并缩放到训练尺寸
    '''
    def __init__(self, output_size=320, crop_ratio=0.9, resize_scales=[0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2], whiten=False):
        assert isinstance(output_size, (int, tuple))
        self.resize_scales = resize_scales
        self.output_size = output_size # 320
        self.crop_ratio = crop_ratio
        self.whiten = whiten

    def __call__(self, sample):
        input_t, target_t = sample['image'], sample['label']
        h, w = input_t.shape[:2]

        resize_scale = random.choice(self.resize_scales)
        resize_h, resize_w = int(h * resize_scale), int(w * resize_scale)
        
        
        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        input_t = cv2.resize(input_t, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
        target_t = cv2.resize(target_t, (resize_w, resize_h), interpolation=cv2.INTER_CUBIC)
        target_t = np.expand_dims(target_t, -1)
        
        if random.random() > 0.5: # 水平反转
            input_t = input_t[:, ::-1, :]
            target_t = target_t[:, ::-1, :]

        if random.random() >= 0.5: # 垂直反转
            input_t = input_t[::-1]
            target_t = target_t[::-1]



        # crop需要方形，以较短的边*0.9作为边长
        crop_s =  resize_w * self.crop_ratio if resize_h > resize_w else resize_h * self.crop_ratio  

        crop_s = int(crop_s)
        h1 = np.random.randint(0, resize_h - crop_s)
        w1 = np.random.randint(0, resize_w - crop_s)

        input_t = input_t[h1: h1 + crop_s, w1: w1 + crop_s, :]
        target_t = target_t[h1: h1 + crop_s, w1: w1 + crop_s, :]

        # 统一到训练尺寸
        input_t = cv2.resize(input_t, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
        target_t = cv2.resize(target_t, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
        if self.whiten:
            target_t[target_t > 80] = 255

        return {'image':input_t, 'label':target_t}



class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, ):
        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.img_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def __call__(self, sample):
        input_t, target_t = sample['image'], sample['label']
        input_t = input_t.astype('float32')
        target_t = target_t.astype('float32')

        input_t = (input_t / 255. - self.img_mean) / self.img_std
        target_t = (target_t / 255.) 

        if len(target_t.shape) < 3:
            target_t = np.expand_dims(target_t, -1)

        input_t = torch.from_numpy(input_t).permute(2, 0, 1).type(torch.float32)
        target_t = torch.from_numpy(target_t).permute(2, 0, 1).type(torch.float32)

        return {'image': input_t, 'label': target_t}

class HumanSegDataset(data.Dataset):
    def __init__(self, root_dir, img_name_list, transform=None):
        self.root_dir = root_dir
        self.image_name_list_txt = img_name_list
        self.transform = transform
        self._prepare_list()

    def _prepare_list(self):
        self.image_name_list = []
        self.label_name_list = []
        train_lines = open(self.image_name_list_txt).readlines()
        train_lines = [l.strip().strip('\n') for l in train_lines]
        for short_path in train_lines:
            # clean_train_images_3rd_an_az/an/anImages/10027846573011.png
            # dirname, image_mask, pngname = short_path.strip().strip('\n').split('/')
            dirname, _, pngname = short_path.rpartition('/') # ('clean_train_images_3rd_an_az/an/anImages', '/', '10027846573011.png')
            self.image_name_list.append(os.path.join(self.root_dir, short_path))
            dirname = dirname.replace('Images', 'Masks')
            self.label_name_list.append(os.path.join(self.root_dir, dirname, pngname[:-4]+'_1.png'))

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image_path = self.image_name_list[idx]
        label_path = self.label_name_list[idx]
        input_t = np.array(Image.open(image_path).convert('RGB'))
        target_t = np.array(Image.open(label_path).convert('L')) # np.max() 255
        # print('input shape: ', input_t.shape) # (800, 800, 3)

        sample = {'image':input_t, 'label':target_t}

        if self.transform:
            sample = self.transform(sample)

        return sample

