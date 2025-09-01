import os
import shutil

# 指定目录路径

# 遍历目录下的所有文件
for i in range(10):
    src_file = 'T{}t0.csv'.format(i + 9)
    dst_file = 'T{}t0.csv'.format(i)
    shutil.move(src_file, dst_file)
    src_file = 'T{}t1.csv'.format(i + 9)
    dst_file = 'T{}t1.csv'.format(i)
    shutil.move(src_file, dst_file)