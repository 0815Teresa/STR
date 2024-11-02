from tqdm.std import tqdm

from ocTree.octree import OctreeIndex


# build tree
def build_tree(traj_data, x_range, y_range, z_range, max_items, max_depth):
    octree = OctreeIndex(bbox=(x_range[0], y_range[0], z_range[0], x_range[1], y_range[1], z_range[1]), max_items=max_items, max_depth=max_depth)
    point_num = 0

    for i in tqdm(range(len(traj_data))):
        for j in range(len(traj_data[i])):
            point_num += 1
            x, y, t = traj_data[i][j]
            # The jth point of the i-th trajectory, labeled for easy recording during later cascade traversal.
            octree.insert(point_num, (x, y, t, i, j, x, y, t, i, j))

    print("traj point nums:", point_num)

    return octree
