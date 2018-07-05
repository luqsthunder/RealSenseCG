import os
from shutil import copytree
import numpy as np

src_folder = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/dynamic_poses/DB/norm_depth'
classes_content = []
higher_samples_cls = 0
for cls_name in sorted(os.listdir(src_folder), key=len):
    cls_name = os.path.join(src_folder, cls_name)
    amount_samples_cls = len(os.listdir(cls_name))
    if amount_samples_cls > higher_samples_cls:
        higher_samples_cls = amount_samples_cls

test_percent = 60
max_samples_pclass = 300
max_k = int(100 / test_percent)

folders_name = 'C:/Users/lucas/Documents/Projects/realsensecg/Gestures/dynamic_poses/Folders4060'

for k in range(max_k):
    curr_folder_name = os.path.join(folders_name, 'F' + str(k + 1))
    print('kfolding for k->{}'.format(k))
    for idx, cls_name in enumerate(sorted(os.listdir(src_folder), key=len)):
        samples_curr_class = sorted(os.listdir(os.path.join(src_folder,
                                                            cls_name)), key=len)
        samples_curr_class = [os.path.join(src_folder, cls_name, l)
                              for l in samples_curr_class]

        spl_curr_cls = np.arange(len(samples_curr_class))
        np.random.seed(5)
        np.random.shuffle(spl_curr_cls)

        test_first_cut = int(max_samples_pclass * test_percent / 100) * k
        test_last_cut = int(max_samples_pclass * test_percent / 100) * (k + 1)

        test = spl_curr_cls[test_first_cut: test_last_cut]
        train = np.concatenate([spl_curr_cls[:test_first_cut],
                                spl_curr_cls[test_last_cut:max_samples_pclass]])

        print('...begin copy train samples ' + cls_name)
        for train_sample in train:
            copytree(samples_curr_class[train_sample],
                     os.path.join(curr_folder_name, 'train', cls_name,
                                  'e' + str(train_sample)))
        print('...finished copy train samples ' + cls_name)
        print('...begin copy test samples ' + cls_name)
        for test_sample in test:
            copytree(samples_curr_class[test_sample],
                     os.path.join(curr_folder_name, 'test', cls_name,
                                  'e' + str(test_sample)))
        print('...finished copy test samples ' + cls_name)
