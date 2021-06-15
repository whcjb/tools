import io
import os
import json
import tqdm
import zlib
import base64
import colorsys
import requests
import numpy as np
from PIL import Image
from six.moves import urllib
from pathlib import Path

from multiprocessing.pool import Pool


num_threads = 20

sql_list = '../lst/0501_0609_.txt'

output_root_dir = Path('../changjing_no_clean/changjing_0510_0607/')

ori_img_dir = output_root_dir / 'ori_images'
os.makedirs(ori_img_dir, exist_ok=True)

trans_image_dir = output_root_dir / 'trans_images'
os.makedirs(trans_image_dir, exist_ok=True)

def split(l, n):
	'''yeild successive n-size chunks from l.'''
	for i in range(0, len(l), n):
		yield l[i: i+n]

def get_url_image_toudi(img_url):
	jfs_url = 'http://img11.360buyimg.local/da/'
	if img_url.startswith('http'):
		im_byte = urllib.request.urlopen(img_url)
	elif img_url.startswith('jfs'):
		im_byte = urllib.request.urlopen(jfs_url + img_url)
	else:
		raise Exception("img_url error.")
	return  Image.open(io.BytesIO(im_byte.read())).convert('RGBA') 

def get_url_image(img_url):
	jfs_url = 'http://img11.360buyimg.local/da/'
	if img_url.startswith('http'):
		im_byte = urllib.request.urlopen(img_url)
	elif img_url.startswith('jfs'):
		im_byte = urllib.request.urlopen(jfs_url + img_url)
	else:
		raise Exception("img_url error.")
	return  Image.open(io.BytesIO(im_byte.read())).convert('RGB')    

def background_detect(img_url):
	headers = {
	'Content-Type': 'application/json',
	}
	data = {"request_type": ["BackgroundDetect"], \
			"img_url":img_url}
	response = requests.post('http://aiservice.jd.local/ai_service', \
		headers=headers, data=json.dumps(data))
	data = response.json()
	print(data)
	if data[0]['ret'] == 1:
		return 1
	return data[0]['result'][0]['result']['result'][0]['bg_type']

def func(args_lst):
	sku, ori_url, trans_url = args_lst
	if os.path.exists(os.path.join(ori_img_dir, sku+'.png')):
		return
	try:
		if background_detect(ori_url) == 0:
			cj_im = get_url_image(ori_url)
			td_im = get_url_image_toudi(trans_url)
			ori_sv_path = os.path.join(ori_img_dir, sku+'.png') 
			trans_sv_path = os.path.join(trans_image_dir, sku+'_1.png') 
			# if not os.path.exists(ori_sv_path):
			cj_im.save(ori_sv_path)
			print('save image, ', ori_sv_path)
			# if not os.path.exists(trans_sv_path):
			td_im.save(trans_sv_path)
			print('save image, ', trans_sv_path)
	except Exception as e:
		print(e)


lines = open(sql_list).readlines()
line_cnt = len(lines)

print('=>url numbers: ', line_cnt)

splits = list(split(lines, num_threads*1000))
for i, nl in enumerate(splits):
	process_lst = []
	for l in nl:
		sku, ori_url, trans_url = l.strip('\n').split('\t')
		process_lst.append([sku, ori_url, trans_url])
	pool = Pool(processes=20)
	for _ in tqdm.tqdm(pool.imap_unordered(func, process_lst), total=len(process_lst)):
	    pass
	

# pool = Pool(processes=20)
# for _ in tqdm.tqdm(pool.imap_unordered(func, process_lst), total=len(process_lst)):
#     pass

# with Pool(2) as pool:
# 	pool.map(func, process_lst)
