""" Collect point clouds and the corresponding labels from original S3DID dataset, and save into numpy files.

Author: Zhao Na, 2020
"""

import os
import glob
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def collect_point_label(anno_path, out_filename, file_format='numpy'):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.
    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
        file_format: txt or numpy, determines what file format to save.
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """
    points_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in CLASS_NAMES:  # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * CLASS2LABEL[cls]
        points_list.append(np.concatenate([points, labels], 1))  # Nx7

    data_label = np.concatenate(points_list, 0)
    # xyz_min = np.amin(data_label, axis=0)[0:3]
    # data_label[:, 0:3] -= xyz_min

    if file_format == 'txt':
        fout = open(out_filename, 'w')
        for i in range(data_label.shape[0]):
            fout.write('%f %f %f %d %d %d %d\n' % \
                       (data_label[i, 0], data_label[i, 1], data_label[i, 2],
                        data_label[i, 3], data_label[i, 4], data_label[i, 5],
                        data_label[i, 6]))
        fout.close()
    elif file_format == 'numpy':
        np.save(out_filename, data_label)
    else:
        print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
              (file_format))
        exit()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version',
                        help='Directory to dataset')
    args = parser.parse_args()


    DATA_PATH = args.data_path
    folders = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
    DST_PATH = os.path.join(ROOT_DIR, 'datasets/S3DIS')
    SAVE_PATH = os.path.join(DST_PATH, 'scenes', 'data')
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    CLASS_NAMES = [x.rstrip() for x in open(os.path.join(ROOT_DIR, 'datasets/S3DIS/meta', 's3dis_classnames.txt'))]
    CLASS2LABEL = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    for folder in folders:
        print("=================\n   " + folder + "\n=================")

        data_folder = os.path.join(DATA_PATH, folder)
        if not os.path.isdir(data_folder):
            raise ValueError("%s does not exist" % data_folder)

        # all the scenes in current Area
        scene_paths = [os.path.join(data_folder, o) for o in os.listdir(data_folder)
                                                if os.path.isdir(os.path.join(data_folder, o))]

        n_scenes = len(scene_paths)
        if (n_scenes == 0):
            raise ValueError('%s is empty' % data_folder)
        else:
            print('%d files are under this folder' % n_scenes)

        for scene_path in scene_paths:
            # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
            anno_path = os.path.join(scene_path, "Annotations")
            print(anno_path)
            elements = scene_path.split('/')
            out_filename = '{}_{}.npy'.format(elements[-2], elements[-1]) # Area_1_hallway_1.npy
            try:
                collect_point_label(anno_path, os.path.join(SAVE_PATH, out_filename))
            except:
                print(anno_path, 'ERROR!!')


