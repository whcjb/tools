import os
import shutil
import imagededup
from imagededup.methods import PHash

import tqdm
from multiprocessing.pool import Pool

num_threads = 20

ori_image = '../changjing_no_clean/changjing_0510_0607/ori_images/'
# ori_image = '/home/yanpeifa/matting/pytorch_latefusion/data/test_image'
ori_img = '../changjing_no_clean/changjing_0510_0607/ori_images/'
trans_img = '../changjing_no_clean/changjing_0510_0607/trans_images/'
trans_img_new = '../trans_img_new/'
align_img = '../align_img/'
align_img_new = '../align_img_new/'

phasher = PHash()

# 生成图像目录中所有图像的二值hash编码
encodings = phasher.encode_images(image_dir=ori_image)

# 对已编码图像寻找重复图像， 返回 list，包含所有重复文件
duplicates = phasher.find_duplicates_to_remove(encoding_map=encodings, max_distance_threshold=3)

print('=> finding %d duplicate images.' % len(duplicates))
print(duplicates)
def func(image_name):
    image_stem = image_name.split('.')[0]
    mask_name = image_stem + '_1.png'
    image_path = os.path.join(ori_image, image_name)
    mask_path = os.path.join(trans_img, mask_name)
    try:
        os.remove(image_path)
        os.remove(mask_path)
    except Exception as e:
        print(e)

pool = Pool(processes=num_threads)
for _ in tqdm.tqdm(pool.imap_unordered(func, duplicates), total=len(duplicates)):
    pass



#  去掉多余图片
# cnt = 0
# for tim in os.listdir(trans_img):
#     if tim.endswith('_1.png'):
#         im_stem = tim.split('_')[0]
#         tim_path = trans_img / tim
#         ori_name = im_stem + '.png'
#         ori_path = ori_img / ori_name
#         if not os.path.exists(ori_path):
#             os.remove(tim_path)
#             cnt += 1
# print(cnt)
