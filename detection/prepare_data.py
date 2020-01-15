import os
import shutil
from glob import glob

python_name = 'python' if os.name == 'nt' else 'python3'
os.system(f'{python_name} egohands_dataset_clean.py')

root = os.getcwd()
os.makedirs(os.path.join(root, 'data', 'Hand', 'images'), exist_ok=True)

for image in glob(os.path.join(root, 'images', 'train', '*.jpg')):
    shutil.move(image, os.path.join(root, 'data', 'Hand', 'images'))
for image in glob(os.path.join(root, 'images', 'test', '*.jpg')):
    shutil.move(image, os.path.join(root, 'data', 'Hand', 'images'))

os.system(f'{python_name} convert_to_voc.py')

for image in glob(os.path.join(root, 'annotation', 'VOC2007', '*')):
    print(image)
    shutil.move(image, os.path.join(root, 'data', 'Hand'))

shutil.rmtree('annotation')
shutil.rmtree('egohands')
shutil.rmtree('images')
