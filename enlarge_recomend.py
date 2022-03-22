import os
import sys
import time
import queue
import shutil
import pickle
import logging
import threading
import collections
from pathlib import Path
from config import _c as config
from PIL import Image, ImageFont

from get_detail_images import *
from detail_page_image_filter_with_threshold import image_filter
from ocr_filter import ocr_filter, ocr_filter_main0im
from salience_filter import *
from similarity_sort import *
from osnet_similarity import osnet_similar_filter_mainim
from metriclearn_similarity import metriclearn_similar_filter_mainim
from utils import *

'''
    1. 检测白底图直接返回
    2. 场景图过滤
'''
buf_size = 1000

num_producer = 10
num_consumer = 10

sku_ims_urls_d = {}
# rst_file = Path('./dataset/mainim4k_url.pkl')
# with open(rst_file, 'rb') as f:
#     sku_ims_urls_d = pickle.load(f)

# logging.basicConfig(level=logging.DEBUG,
#                     format='(%(threadName)-9s) %(message)s',)


class ProduceImagesThread(threading.Thread):
    def __init__(self, name, skus_deque, skus_done_queue, output_dir):
        super(ProduceImagesThread, self).__init__()
        print('start producer {} ...'.format(name))
        # lines = open(skus_lst).readlines()
        # self.skus = [line.strip('\n') for line in lines]
        self.seeker = MaterialSeeker()
        # self.material_types = [1, 2] # 2详情图
        self.material_types = [1,]
        self.output_dir = output_dir
        self.skus_deque = skus_deque
        self.skus_done_queue = skus_done_queue
        self.dict_loct = threading.Lock()

    def run(self):
        while self.skus_deque:
            sku_str = self.skus_deque.pop()
            has_im = self.get_sku_images(sku_str)
            if not self.skus_done_queue.full() and has_im:
                self.skus_done_queue.put(sku_str)
        return

    def get_sku_images(self, sku_str):
        print('processing {} ...'.format(sku_str))
        has_im = False
        for material_type in self.material_types:
            try:
                sku_dir = out_image_dir / sku_str
                try: # 获取透底图
                    trans_url = get_trans_url(sku_str)
                    trans_img = get_url_image_no_convert(trans_url)
                    trans_img.save(sku_dir/(sku_str+'_trans.png'))
                    print('success get trans image.', sku_dir)
                except Exception as e:
                    print('get trans image failed.', sku_dir)

                # task_type, data = self.seeker.get_sku_materials(sku_str, material_type=material_type)
                task_type, data = self.seeker.get_mainim_liannu(sku_str)
                urls = data['image_urls']
                if material_type == 1:
                    ctype = 'main'
                    urls_cut = []
                    for url in urls:
                        if url.startswith('http'):
                            url = 'jfs' + url.split('jfs')[1]
                        urls_cut.append(url)
                elif material_type == 2:
                    ctype = 'detail'
                    urls_cut = urls
                else:
                    ctype = ''
                
                os.makedirs(sku_dir, exist_ok=True)
                for i, url in enumerate(urls_cut):
                    im_s = sku_str+'_'+ctype+'_'+str(i)+'.png'
                    sv_path = sku_dir / im_s
                    
                    if os.path.exists(sv_path):
                        continue

                    with self.dict_loct:
                        if not im_s in sku_ims_urls_d:
                            sku_ims_urls_d[im_s] = url
                    img = get_url_image_no_convert(url)
                    img.save(sv_path)
                    
                    has_im = True
                # print('get images done ... ', sku_str)
            except Exception as e:
                print('get_sku_images.', e)
                continue
        return has_im

class PutRecomendThread(threading.Thread):
    def __init__(self, name, config, image_root_dir, skus_done_queue, deque_buf_size=50):
        super(PutRecomendThread, self).__init__()
        print('start PutRecomendThread {}'.format(name))
        self.image_root_dir = image_root_dir
        self.buffer_deque = collections.deque(maxlen=deque_buf_size)
        self.skus_done_queue = skus_done_queue
        
        self.push_cdn = config.push_cdn
        self.salience_cfg = config.salience_cfg
        self.coarsecut_cfg = config.coarsecut_cfg
        self.ocr_cfg = config.ocr_cfg
        self.similarity_cfg = config.similarity_cfg
        self.font = ImageFont.truetype("./font/yaheibd_mono.ttf", 24)
        self.rec_obj_tables_index = config.rec_obj_index_tables
        with open(self.rec_obj_tables_index, 'rb') as handle:
            self.rec_feats = pickle.load(handle)
        self.rec_feats_norm = np.sqrt(np.sum(self.rec_feats**2, axis=1, keepdims=True)) # (10, 1792)

    def run(self):
        while True:
            if not self.skus_done_queue.empty():
                sku_str = self.skus_done_queue.get()
                print('Getting ' + str(sku_str) + ' : ' + str(skus_done_queue.qsize()) + ' items in queue')
                sku_dir = self.image_root_dir / sku_str
                # try:
                # shutil.rmtree(sku_dir)
                # print('rm {} ...'.format(sku_dir))
                # # except Exception as e:
                #     print(e)
                # try:
                self.buffer_deque.clear()
                self.filter(sku_dir, self.coarsecut_cfg)
                time.sleep(1)
                # except Exception as e:
                #     print(e)
        return

    def filter(self, sku_dir, cfg):
        # =====================
        # coarse_filter白边双图宽图高图矩形抠图过滤
        # =====================
        for im in os.listdir(sku_dir):
            if 'thresh' in im:
                continue
            if (im.endswith('.png') or im.endswith('.jpg')) and 'detail' in im: # 只过滤商详图
                im_path = sku_dir / im
                # try:
                image_filter(sku_dir, str(im_path), self.buffer_deque, cfg)
                # except Exception as e:
                #     print(e)
            if 'main' in im: # 商品图不过滤
                im_path = sku_dir / im
                pimg = Image.open(im_path).convert('RGB')
                self.buffer_deque.appendleft([str(im_path), pimg])
                if not self.salience_cfg.visualize:
                    os.remove(im_path)

            if 'trans' in im:
                im_path = sku_dir / im
                pimg = Image.open(im_path).convert('RGB')
                self.buffer_deque.appendleft([str(im_path), pimg])

        self.buffer_deque.appendleft('')
        # =======================
        # salience_cut显著性区域切割
        # =======================
        while self.buffer_deque:
            item = self.buffer_deque.pop()
            if item == '':
                break
            im_path, pimg = item
            bname = im_path.split('/')[-1]
            w, h = pimg.size
            if 'trans' in bname:
                self.buffer_deque.appendleft([im_path, pimg, np.ones([h, w])])
                continue
            if 'main' in bname and w == h:
                is_val, mask_np = salience_detect(im_path, pimg, self.salience_cfg)
                if 'main_0' in bname:
                    self.buffer_deque.appendleft([im_path, pimg, mask_np])
                    continue
                else:
                    if is_val:
                        self.buffer_deque.appendleft([im_path, pimg, mask_np])
                        continue
                    else:
                        if self.salience_cfg.visualize:
                            pimg.save(im_path[:-4]+'nosal.png')
                        continue
            else: # 详情图切割
                has_salience, crop_im, is_cut, crop_mask_np = salience_cut(im_path, pimg, self.salience_cfg)
                if has_salience or 'main_0' in bname:
                    # if is_cut:
                    # im_path = str(im_path)[:-4]+'_cube.png' # 详情图使用
                    # else:
                    #     im_path = str(im_path)[:-4]+'_raw.png'
                    self.buffer_deque.appendleft([im_path, crop_im, crop_mask_np])
        self.buffer_deque.appendleft('')

        # print('============salience_filter')
        # for item in self.buffer_deque:
        #     if item == '':
        #         continue
        #     im_path, pimg, mask_np = item
        #     print(im_path)
        #     pimg.save(im_path)
        # print('========================\n')

        # =========================
        # ocr filter, 文字过多删除
        # =========================
        while self.buffer_deque:
            item = self.buffer_deque.pop()
            if item == '': # filter隔板   
                break
            im_path, pimg, mask_np = item
            basename = im_path.split('/')[-1]

            if 'trans' in basename:
                self.buffer_deque.appendleft([im_path, pimg, mask_np, [], False])
                continue

            # if 'main_0' in basename: # 商品图第一张不过滤
            #     rst = ocr_filter_main0im(pimg)
            #     self.buffer_deque.appendleft([im_path, pimg, mask_np, rst])
                # continue  

            # 拼图过滤, 矩形过滤
            isrec, max_rec_aera_ratio = is_rectangle_mask(im_path, pimg, mask_np, self.salience_cfg, self.rec_feats, self.rec_feats_norm)
            # if isrec:
            #     if self.salience_cfg.visualize:
            #         pimg.save(im_path[:-4]+'_hasrec.png')
            #     continue
            
            # 文字过滤
            is_ocr_good, comman_rst = ocr_filter(pimg, mask_np, max_rec_aera_ratio, ocr_line_thresh=5, im_path=im_path, cfg=self.ocr_cfg) # 合格的循环滚入，不合格的自动丢弃
            self.buffer_deque.appendleft([im_path, pimg, mask_np, comman_rst, is_ocr_good])
            if is_ocr_good:
                pass
                # 复制矩形先显著性图到
                # if max_rec_aera_ratio > 0.977 and isrec:
                #     im_name = im_path.split('/')[-1][:-4] + '_' + str(round(max_rec_aera_ratio, 3)) + '.png' 
                #     dst_path = os.path.join(self.salience_cfg.tmp_rec_dir, im_name)
                #     mask_im = Image.fromarray(((mask_np > 220)*255).astype(np.uint8)).convert('RGB')
                #     mask_im.save(dst_path[:-4]+'_maskrec.png')
                #     shutil.copy(im_path, dst_path)
            else:
                if self.ocr_cfg.visualize:
                    im_path = im_path[:-4]+'_ocrbad.png'
                    pimg.save(im_path)
                    # self.buffer_deque.appendleft([im_path, pimg, mask_np])

        self.buffer_deque.appendleft('')
        
        # print('============ocr_filter')
        # for item in self.buffer_deque:
        #     if item == '':
        #         continue
        #     im_path, pimg, _, _ = item
        #     print(im_path)
        #     pimg.save(im_path)
        # print('========================\n')
        
        # # 相似性过滤
        # # print('=========similarity_ims: ', self.buffer_deque)
        if self.similarity_cfg.similarity_type == 'imagenet':
            is_sorted, similarity_ims, _ = similarity_filter_buffer(self.buffer_deque, 
                        self.similarity_cfg)
            get_valid_img_according_to_th(similarity_ims, is_sorted, 
                        sku_ims_urls_d, push_cdn=self.push_cdn, 
                        put_text=self.similarity_cfg.put_text, font=self.font)
        elif self.similarity_cfg.similarity_type == 'osnet':
            similarity_ims = osnet_similar_filter_mainim(self.buffer_deque, 
                        self.similarity_cfg)
            get_valid_img_according_to_osnet(similarity_ims, sku_ims_urls_d, 
                        push_cdn=self.push_cdn, put_text=self.similarity_cfg.put_text, 
                        font=self.font)
        elif self.similarity_cfg.similarity_type == 'metriclearn':
            similarity_ims = metriclearn_similar_filter_mainim(self.buffer_deque, self.similarity_cfg)
            get_valid_img_according_to_metriclearn(similarity_ims, sku_ims_urls_d, 
                        push_cdn=self.push_cdn, put_text=self.similarity_cfg.put_text, 
                        font=self.font)
        else:
            raise NotImplentError

if __name__ == '__main__':
    '''
        usage:
        python enlarge_recomend_images_multithread.py skus.lst
    '''
    skus_lst = sys.argv[1]

    skus_done_queue = queue.Queue(buf_size)
    out_image_dir = Path('./content_img/') # 图像保存目录
    # out_image_dir = Path('./dataset/mainim4k')

    lines = open(skus_lst).readlines()
    sku_lst = [line.strip().strip('\n') for line in lines]
    sku_deque = collections.deque(sku_lst)
    producers = []
    producer_tasks = []
    for pc in range(num_producer):
        name = 'producer-{}'.format(pc)
        p = ProduceImagesThread(name=name, skus_deque=sku_deque, skus_done_queue=skus_done_queue, output_dir=out_image_dir)
        producer_tasks.append(p)
        p.start()
        
    for cn in range(num_consumer):
        name = 'consumer-{}'.format(cn)
        c = PutRecomendThread(name=name, config=config, skus_done_queue=skus_done_queue, image_root_dir=out_image_dir)
        c.start()

    for p in producer_tasks:
        p.join()

    rst_file  = skus_lst.split('.')[0]+'.pkl'

    with open(rst_file, 'wb') as f:
        pickle.dump(sku_ims_urls_d, f, protocol=pickle.HIGHEST_PROTOCOL)


    # name = 'consumer-{}'.format(1)
    # c = PutRecomendThread(name=name, config=config, skus_done_queue=skus_done_queue, image_root_dir=out_image_dir)
    # c.start()

