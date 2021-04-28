""" Data Preprocess and Loader for ScanNetV2 Dataset

Author: Zhao Na, 2020
"""
import os
import glob
import numpy as np
import pickle


class ScanNetDataset(object):
    def __init__(self, cvfold, data_path):
        self.data_path = data_path
        self.classes = 21
        # self.class2type = {0:'unannotated', 1:'wall', 2:'floor', 3:'chair', 4:'table', 5:'desk', 6:'bed', 7:'bookshelf',
        #                    8:'sofa', 9:'sink', 10:'bathtub', 11:'toilet', 12:'curtain', 13:'counter', 14:'door',
        #                    15:'window', 16:'shower curtain', 17:'refridgerator', 18:'picture', 19:'cabinet', 20:'otherfurniture'}
        class_names = open(os.path.join(os.path.dirname(data_path), 'meta', 'scannet_classnames.txt')).readlines()
        self.class2type = {i: name.strip() for i, name in enumerate(class_names)}
        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.types = self.type2class.keys()

        self.fold_0 = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair','counter', 'curtain', 'desk', 'door', 'floor']
        self.fold_1 = ['otherfurniture', 'picture', 'refridgerator', 'shower curtain', 'sink', 'sofa', 'table', 'toilet', 'wall', 'window']

        if cvfold == 0:
            self.test_classes = [self.type2class[i] for i in self.fold_0]
        elif cvfold == 1:
            self.test_classes = [self.type2class[i] for i in self.fold_1]
        else:
            raise NotImplementedError('Unknown cvfold (%s). [Options: 0,1]' %cvfold)

        all_classes = [i for i in range(1, self.classes)]
        self.train_classes = [c for c in all_classes if c not in self.test_classes]

        self.class2scans = self.get_class2scans()

    def get_class2scans(self):
        class2scans_file = os.path.join(self.data_path, 'class2scans.pkl')
        if os.path.exists(class2scans_file):
            #load class2scans (dictionary)
            with open(class2scans_file, 'rb') as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = .05  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k:[] for k in range(self.classes)}

            for file in glob.glob(os.path.join(self.data_path, 'data', '*.npy')):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:,6].astype(np.int)
                classes = np.unique(labels)
                print('{0} | shape: {1} | classes: {2}'.format(scan_name, data.shape, list(classes)))
                for class_id in classes:
                    #if the number of points for the target class is too few, do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0]*min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print('==== class to scans mapping is done ====')
            for class_id in range(self.classes):
                print('\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}'.format(
                          class_id,  min_ratio, min_pts, self.class2type[class_id], len(class2scans[class_id])))

            with open(class2scans_file, 'wb') as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pre-training on ShapeNet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options: {0,1}')
    parser.add_argument('--data_path', type=str, default='../datasets/ScanNet/blocks_bs1_s1', help='Directory to source data')
    args = parser.parse_args()
    dataset = ScanNetDataset(args.cvfold, args.data_path)