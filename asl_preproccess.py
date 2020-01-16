import os
import random

os.chdir('asl_alphabet')

test_dir = os.path.abspath('test')

if os.path.exists(os.path.join(test_dir, 'asl_alphabet_test')):
    for filename in os.listdir(os.path.join(test_dir, 'asl_alphabet_test')):
        letter_dir = os.path.join(test_dir, filename[:filename.index('_')])

        if not os.path.exists(letter_dir):
            os.mkdir(letter_dir)

        if not os.path.exists(os.path.join(letter_dir, filename)):
            os.rename(os.path.join(test_dir, 'asl_alphabet_test', filename),
                        os.path.join(letter_dir, filename))

    os.rmdir(os.path.join(test_dir, 'asl_alphabet_test'))

del_folder = os.path.join(test_dir, 'del')
if not os.path.exists(del_folder):
    os.mkdir(del_folder)

random.seed(0)
imgs_idxs = random.sample(range(1, 3001), 100)

train_dir = os.path.abspath('train')

for dirname in os.listdir(train_dir):
    for idx in imgs_idxs:
        letter_path = os.path.join(train_dir, dirname, f'{dirname}{idx}.jpg')
        new_letter_path = os.path.join(test_dir, dirname, f'{dirname}{idx}.jpg')
        if os.path.exists(letter_path) and not os.path.exists(new_letter_path):
            os.rename(letter_path, new_letter_path)
