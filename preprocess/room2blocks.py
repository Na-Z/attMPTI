""" Processing step 1, split room into blocks

Author: Zhao Na, 2020
"""

import os
import glob
import numpy as np

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR SUPERPOINT GRAPH GENERATION
# -----------------------------------------------------------------------------

def room2blocks(data, block_size, stride, min_npts):
    """ Prepare block data.
    Args:
        data: N x 7 numpy array, 012 are XYZ in meters, 345 are RGB in [0,255], 6 is the labels
            assumes the data is not shifted (min point is not origin),
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
    Returns:
        blocks_list: a list of blocks, each block is a num_point x 7 np array
    """
    assert (stride <= block_size)

    xyz = data[:,:3]
    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    xyz_max = np.amax(xyz, axis=0)

    # Get the corner location for our sampling blocks
    xbeg_list = []
    ybeg_list = []
    num_block_x = int(np.ceil((xyz_max[0] - block_size) / stride)) + 1
    num_block_y = int(np.ceil((xyz_max[1] - block_size) / stride)) + 1
    for i in range(num_block_x):
        for j in range(num_block_y):
            xbeg_list.append(i * stride)
            ybeg_list.append(j * stride)

    # Collect blocks
    blocks_list = []
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        xcond = (xyz[:, 0] <= xbeg + block_size) & (xyz[:, 0] >= xbeg)
        ycond = (xyz[:, 1] <= ybeg + block_size) & (xyz[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < min_npts:  # discard block if there are less than 100 pts.
            continue

        block = data[cond, :]
        blocks_list.append(block)

    return blocks_list


def room2blocks_wrapper(room_path, block_size, stride, min_npts):
    if room_path[-3:] == 'txt':
        data = np.loadtxt(room_path)
    elif room_path[-3:] == 'npy':
        data = np.load(room_path)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks(data, block_size, stride, min_npts)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='[Preprocessing] Split rooms into blocks')
    parser.add_argument('--data_path', default='../datasets/S3DIS/scenes')
    parser.add_argument('--dataset', default='s3dis', metavar='bs', help='s3dis|scannet')
    parser.add_argument('--block_size', type=float, default=1, metavar='s', help='size of each block')
    parser.add_argument('--stride', type=float, default=1, help='stride of sliding window for splitting rooms, '
                                                                'stride should be not larger than block size')
    parser.add_argument('--min_npts', type=int, default=1000, help='the minimum number of points in a block,'
                                                                  'if less than this threshold, the block is discarded')

    args = parser.parse_args()

    DATA_PATH = args.data_path
    BLOCK_SIZE = args.block_size
    STRIDE = args.stride
    MIN_NPTS = args.min_npts
    SAVE_PATH = os.path.join(os.path.dirname(DATA_PATH), 'blocks_bs{0}_s{1}'.format(BLOCK_SIZE, STRIDE), 'data')
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    file_paths = glob.glob(os.path.join(DATA_PATH, 'data', '*.npy'))
    print('{} scenes to be split...'.format(len(file_paths)))

    block_cnt = 0
    for file_path in file_paths:
        room_name = os.path.basename(file_path)[:-4]
        blocks_list = room2blocks_wrapper(file_path, block_size=BLOCK_SIZE, stride=STRIDE, min_npts=MIN_NPTS)
        print('{0} is split into {1} blocks.'.format(room_name, len(blocks_list)))
        block_cnt += len(blocks_list)

        for i, block_data in enumerate(blocks_list):
            block_filename = room_name + '_block_' + str(i) + '.npy'
            np.save(os.path.join(SAVE_PATH, block_filename), block_data)

    print("Total samples: {0}".format(block_cnt))