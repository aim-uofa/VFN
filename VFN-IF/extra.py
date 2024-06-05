import os
import shutil
import json

def copyfile(src_dir, dest_dir, file_name):
    src_file = os.path.join(src_dir, file_name)
    dest_file = os.path.join(dest_dir, file_name)
    # 尝试复制文件，如果出错则跳过
    try:
        shutil.copyfile(src_file, dest_file)
    except Exception as e:
        print(f'Failed to copy file {file_name}. Error: {e}')


json_file = './dataset/json/CATH4.2/train_multi_label.json'
data = json.load(open(json_file, 'r'))

# 源文件夹和目标文件夹
src_dir = '/mnt/nas/datasets/protein/unifold/processed'
dest_dir = './processed/pdb_labels'

# 确保目标目录存在
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 遍历JSON数据中的所有键
for key, values in data.items():
    # 构建文件的相对路径
    for file_name in values:
        # label_name = file_name+'.label.pkl.gz'
        # feature_name = file_name+'.feature.pkl.gz'
        uniprots = file_name+'.label.pkl.gz'
        print(f'Processing file {file_name}...')
        
        # 复制文件
        # copyfile(src_dir, dest_dir, label_name)
        # copyfile(src_dir, dest_dir, feature_name)
        copyfile(src_dir, dest_dir, uniprots)
    

print('File extraction complete.')