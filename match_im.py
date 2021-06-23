import os
import cv2
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from align import *

import tqdm
from multiprocessing.pool import Pool

sift = cv2.xfeatures2d.SIFT_create()

im_lst = '/home/yanpeifa/matting/dataset/matchimg_img/changjing_no_clean/changjing_0510_0607/dedup_0501_0609.lst'
image_dir = './changjing/'

ori_im_dir = Path('/home/yanpeifa/matting/dataset/matchimg_img/changjing_no_clean/changjing_0510_0607/ori_images/')
trans_im_dir = Path('/home/yanpeifa/matting/dataset/matchimg_img/changjing_no_clean/changjing_0510_0607/trans_images/')

output = Path('/home/yanpeifa/matting/dataset/matchimg_img/ori_trans_align/data0501_0609')
new_or_dir = output / 'images'
os.makedirs(new_or_dir, exist_ok=True)

new_tr_dir = output / 'trans'
os.makedirs(new_tr_dir, exist_ok=True)

new_align_dir = output / 'align'
os.makedirs(new_align_dir, exist_ok=True)


num_threads = 20

ratio = 0.3
iter_num = 2000
fit_pos_cnt_thresh = 30


lines = open(im_lst).readlines()
cnt = len(lines)

skus = []

for i, l in enumerate(lines):
    line = l.strip().strip('\n')[2:] # ./122048.png
    if '.png' in line:
        sku = line.split('.')[0]
        skus.append(sku)

    if i % 10000 == 0:
        print('%d / %d' % (i, cnt))


def func(img_stem):
    or_im_path = ori_im_dir / (img_stem + '.png')
    tr_im_path = trans_im_dir / (img_stem + '_1.png')
    if not os.path.exists(or_im_path) or not os.path.exists(tr_im_path):
        return

    try:
        target_im = np.array(Image.open(or_im_path))
        source_im = np.array(Image.open(tr_im_path))
        source_rgb = source_im[:, :, :3]
        a = source_im[:, :, 3]
        a3 = np.stack([a, a, a], axis=-1)
    
        kp_s, desc_s = extract_sift(source_rgb)
        kp_t, desc_t = extract_sift(target_im)
        fit_pos = match_sift(desc_s, desc_t)
        
        if fit_pos.shape[0] < fit_pos_cnt_thresh:
            os.remove(or_im_path)
            os.remove(tr_im_path)
            return
        
        m = affine_matrix(kp_s, kp_t, fit_pos)
        merge, warp_rgb, source = warp_image(source_rgb, target_im, m)
        am, warp_a, _  = warp_image(a3, target_im, m)
        
        warp_a = warp_a[:, :, 0]
        warp_a = np.expand_dims(warp_a, axis=-1)
        merge = np.concatenate((warp_rgb, warp_a), axis=2)
        
        sku = img_stem
        save_path = new_align_dir / (sku + '_2.png')
        or_new_path = new_or_dir / (img_stem + '.png')
        tr_new_path = new_tr_dir / (img_stem + '_1.png')
        shutil.move(or_im_path, or_new_path)
        shutil.move(tr_im_path, tr_new_path)
        Image.fromarray(merge.astype(np.uint8)).save(save_path)
        # print('precess %s' % save_path)
    except Exception as e:
        print(e)

pool = Pool(processes=num_threads)
for _ in tqdm.tqdm(pool.imap_unordered(func, skus), total=len(skus)):
    pass


# for i, l in enumerate(lines):
#     line = l.strip().strip('\n')[13:]
#     if '_' in line:
#         sku = line.split('_')[0]
#         or_im_path = sku + '.png'
#         tr_im_path = sku + '_1.png'
        
#         ori_ims.append(or_im_path)
#         trans_ims.append(tr_im_path)
#     if i % 10000 == 0:
#         print('%d / %d' % (i, cnt))
    
# for or_im, tr_im in zip(ori_ims, trans_ims):
#     or_im_path = image_dir + or_im
#     tr_im_path = image_dir + tr_im
#     if not os.path.exists(or_im_path) or not os.path.exists(tr_im_path):
# #         print('not found.')
#         continue
#     target_im = np.array(Image.open(or_im_path))
#     source_im = np.array(Image.open(tr_im_path))
#     source_rgb = source_im[:, :, :3]
#     a = source_im[:, :, 3]
#     a3 = np.stack([a, a, a], axis=-1)
    
#     kp_s, desc_s = extract_sift(source_rgb)
#     kp_t, desc_t = extract_sift(target_im)
#     fit_pos = match_sift(desc_s, desc_t)
    
#     print(fit_pos.shape[0])
#     if fit_pos.shape[0] < fit_pos_cnt_thresh:
#         print(fit_pos.shape[0])
#         os.remove(or_im_path)
#         os.remove(tr_im_path)
#         print('not same image')
#         continue
    
#     m = affine_matrix(kp_s, kp_t, fit_pos)
#     merge, warp_rgb, source = warp_image(source_rgb, target_im, m)
#     am, warp_a, _  = warp_image(a3, target_im, m)
    
#     warp_a = warp_a[:, :, 0]
#     warp_a = np.expand_dims(warp_a, axis=-1)
#     merge = np.concatenate((warp_rgb, warp_a), axis=2)
    
#     sku = or_im.split('.')[0]
#     save_path = output + sku + '_2.png'
#     or_new_path = output + or_im
#     tr_new_path = output + tr_im
#     shutil.move(or_im_path, or_new_path)
#     shutil.move(tr_im_path, tr_new_path)
#     Image.fromarray(merge.astype(np.uint8)).save(save_path)
#     print('precess %s' % save_path)
    
    
