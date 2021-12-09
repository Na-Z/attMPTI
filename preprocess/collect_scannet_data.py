""" Collect point clouds and the corresponding labels from original ScanNetV2 dataset, and save into numpy files.

Author: Zhao Na, 2020
"""
import os
import sys
import json
import numpy as np
from plyfile import PlyData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def get_raw2scannet_label_map(label_mapping_file):
    lines = [line.rstrip() for line in open(label_mapping_file)]
    lines = lines[1:]
    raw2scannet = {}
    label_classes_set = set(CLASS_NAMES)
    for i in range(len(lines)):
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet


def read_ply_xyzrgb(filename):
    """ read XYZRGB point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices


def collect_point_label(scene_path, scene_name, out_filename):
    # Over-segmented segments: maps from segment to vertex/point IDs
    mesh_seg_filename = os.path.join(scene_path, '%s_vh_clean_2.0.010000.segs.json' % (scene_name))
    # print mesh_seg_filename
    with open(mesh_seg_filename) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
        # print len(seg)
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    # Raw points in XYZRGBA
    ply_filename = os.path.join(scene_path, '%s_vh_clean_2.ply' % (scene_name))
    points = read_ply_xyzrgb(ply_filename)
    print('{0}: {1} points'.format(scene_name, points.shape[0]))

    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    annotation_filename = os.path.join(scene_path, '%s.aggregation.json' % (scene_name))
    # print annotation_filename
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])

    # print len(instance_segids)
    # print labels

    # Each instance's points
    instance_points_list = []
    # instance_labels_list = []
    semantic_labels_list = []
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_points = points[np.array(pointids), :]
        instance_points_list.append(instance_points)
        # instance_labels_list.append(np.ones((instance_points.shape[0], 1)) * i)
        if labels[i] not in RAW2SCANNET:
            label = 'unannotated'
        else:
            label = RAW2SCANNET[labels[i]]
        label = CLASS_NAMES.index(label)
        semantic_labels_list.append(np.ones((instance_points.shape[0], 1)) * label)

    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    scene_points = scene_points[:, 0:6]  # XYZRGB, disregarding the A
    # instance_labels = np.concatenate(instance_labels_list, 0)
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    # data = np.concatenate((scene_points, instance_labels, semantic_labels), 1)
    data = np.concatenate((scene_points, semantic_labels), 1)
    np.save(out_filename, data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/ScanNet/scans',
                        help='Directory to dataset')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    DST_PATH = os.path.join(ROOT_DIR, 'datasets/ScanNet')
    SAVE_PATH = os.path.join(DST_PATH, 'scenes', 'data')
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    meta_path = os.path.join(DST_PATH, 'meta')
    CLASS_NAMES = [x.rstrip() for x in open(os.path.join(meta_path, 'scannet_classnames.txt'))]
    label_mapping_file = os.path.join(meta_path, 'scannetv2-labels.combined.tsv')
    RAW2SCANNET = get_raw2scannet_label_map(label_mapping_file)


    scene_paths = [os.path.join(DATA_PATH, o) for o in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, o))]

    n_scenes = len(scene_paths)
    if (n_scenes == 0):
        raise ValueError('%s is empty' % DATA_PATH)
    else:
        print('%d scenes to be processed...' % n_scenes)

    for scene_path in scene_paths:
        scene_name = os.path.basename(scene_path)
        try:
            out_filename = scene_name+'.npy' # scene0000_00.npy
            collect_point_label(scene_path, scene_name, os.path.join(SAVE_PATH, out_filename))
        except:
            raise ValueError('ERROR {}!!'.format(scene_path))
