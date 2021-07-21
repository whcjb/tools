import os
import sys
import cv2
import tqdm 
from PIL import Image
import subprocess as sp
from pathlib import Path

from multiprocessing import Pool
import tqdm

'''
  调用 denseflow抽光流，多线程执行，每个 class_name 设置一个文件夹
  python dense_flow.py /media/cfs/yanpeifa/action/dataset/kinetics400/3.txt
'''


output_dir = Path('./actions_flow/')
# output_dir = Path('./ceshi/')

os.chdir(output_dir)

print('argv[1]: ', sys.argv[1])
file_name = sys.argv[1]
print('file_name: ', file_name)
lines = open(file_name).readlines()

task_lst = []
for l in lines:
	vid_path = l.strip().strip('\n')
	print(vid_path)
	task_lst.append(vid_path)
print('=> task num: ', len(task_lst))


def func(vid_path):
    # 这个相当于把列表都提取出来了，每一个都是一个视频地址，因此无法设置 class_dir
    # denseflow videolist.txt -b=20 -a=tvl1 -s=1 -cf -v
    class_name = vid_path.split('/')[7] # ['', 'home', 'yanpeifa', 'action', 'dataset', 'kinetics400', 'actions_mp4', 'news_anchoring', 'd6lImsnwfgE_000001_000011.mp4']
    os.makedirs(class_name, exist_ok=True)
    # os.chdir(class_name)
    
    command ='cd ' + class_name + '; pwd; denseflow ' + vid_path + ' -b=20 -a=tvl1 -s=1 -cf -v; cd ..; pwd'
    # command ='denseflow ' + vid_path + ' -b=20 -a=tvl1 -s=1 --cf -v'


    # sp.run(command, stdout=sp.PIPE, shell=True)
    sp.run(command, shell=True)



pool = Pool(processes=8)
for _ in tqdm.tqdm(pool.imap_unordered(func, task_lst), total=len(task_lst)):
    pass
