from operator import itemgetter
import pandas as pd


def _normalize_rect(rect):
    if len(rect) == 5:
        # j-th point of the i-th trajectory
        x1, y1, z1, i1, j1 = rect
        x2, y2, z2, i2, j2 = rect
    else:
        x1, y1, z1, i1, j1, x2, y2, z2, i2, j2 = rect
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if z1 > z2:
        z1, z2 = z2, z1
    return (x1, y1, z1, i1, j1, x2, y2, z2, i2, j2)


def _loopallchildren(parent):
    for child in parent.children:
        if child.children:
            for subchild in _loopallchildren(child):
                yield subchild
        yield child


def get_octree_feat(root, traj_num, traj_size):
    # Initially add the root node, tier, and index at the current tier to the queue
    queue = [(root, 0, 0, 0, 0)] 
    # Record the hierarchy of each leaf node, the index at the hierarchy level; the hierarchical index of the leaf node of the octree in which the jth point of the ith trajectory is located, the hierarchical index of the parent node
    treeid_list_list = [[0]*traj_size for _ in range(int(float(traj_num)))] 

    leaf_level_index = [] 
    parent_level_index = []  

    while queue:
        tree, level, index, parent_level, parent_index = queue.pop(0)

        if len(tree.children) == 0:
            if len(tree.nodes) == 0:
                continue
            else:
                for tp in tree.nodes:
                    tp_rect = tp.rect
                    parent_level_index.append((parent_level, parent_index)) 
                    leaf_level_index.append((level, index))  
                    # (i-th trajectory, j-th point) of the hierarchy
                    # Record the hierarchy of each leaf node, its index in the hierarchy, the hierarchy of the parent node, the index of the parent node in the hierarchy
                    treeid_list_list[tp_rect[3]][tp_rect[4]] = [level, index, parent_level, parent_index] 

        if tree.children:
            for i, child in enumerate(tree.children):
                if child is not None:
                    queue.append((child, level+1, i+8*index, level, index))
    
    # Calculate the maximum and minimum values of the leaf node and parent node hierarchical indexes
    leaf_level_min = min(leaf_level_index, key=lambda x: x[0])
    leaf_level_max = max(leaf_level_index, key=lambda x: x[0])
    leaf_index_min = min(leaf_level_index, key=lambda x: x[1])
    leaf_index_max = max(leaf_level_index, key=lambda x: x[1])

    parent_level_min = min(parent_level_index, key=lambda x: x[0])
    parent_level_max = max(parent_level_index, key=lambda x: x[0])
    parent_index_min = min(parent_level_index, key=lambda x: x[1])
    parent_index_max = max(parent_level_index, key=lambda x: x[1])

    treeid_range = [leaf_level_min[0], leaf_level_max[0], leaf_index_min[1], leaf_index_max[1],
                    parent_level_min[0], parent_level_max[0], parent_index_min[1], parent_index_max[1]]

    return treeid_range, treeid_list_list


class _OctNode(object):
    def __init__(self, item, rect):
        self.item = item
        self.rect = rect

    def __eq__(self, other):
        return self.item == other.item and self.rect == other.rect

    def __hash__(self):
        return hash(self.item)


class _OctTree(object):
    def __init__(self, x, y, z, width, height, depth, max_items, max_depth, _depth=0):
        self.nodes = []
        self.children = []
        self.center = (x, y, z)
        self.width, self.height, self.depth = width, height, depth
        self.max_items = max_items
        self.max_depth = max_depth
        self._depth = _depth

    def __iter__(self):
        for child in _loopallchildren(self):
            yield child

    def _insert(self, item, bbox):
        rect = _normalize_rect(bbox)
        if len(self.children) == 0:
            node = _OctNode(item, rect)
            self.nodes.append(node)

            if len(self.nodes) > self.max_items and self._depth < self.max_depth:
                self._split()
        else:
            self._insert_into_children(item, rect)

    def _remove(self, item, bbox):
        rect = _normalize_rect(bbox)
        if len(self.children) == 0:
            node = _OctNode(item, rect)
            self.nodes.remove(node)
        else:
            self._remove_from_children(item, rect)

    def _intersect_node(self, rect, results=None, uniq=None):
        if results is None:
            rect = _normalize_rect(rect)
            results = []
            uniq = set()
        # search children
        if self.children:
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[0]._intersect_node(rect, results, uniq)
                    else:
                        self.children[4]._intersect_node(rect, results, uniq)
                else:
                    if rect[2] <= self.center[2]:
                        self.children[2]._intersect_node(rect, results, uniq)
                    else:
                        self.children[6]._intersect_node(rect, results, uniq)
            else:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[1]._intersect_node(rect, results, uniq)
                    else:
                        self.children[5]._intersect_node(rect, results, uniq)
                else:
                    if rect[2] <= self.center[2]:
                        self.children[3]._intersect_node(rect, results, uniq)
                    else:
                        self.children[7]._intersect_node(rect, results, uniq)
        # search node at this level
        for node in self.nodes:
            _id = id(node.item)
            if _id not in uniq and node.rect[5] >= rect[0] and node.rect[0] <= rect[5] \
                    and node.rect[6] >= rect[1] and node.rect[1] <= rect[6] \
                    and node.rect[7] >= rect[2] and node.rect[2] <= rect[7]:
                results.append(node.item)
                uniq.add(_id)
        return results
    
    def _intersect_tree(self, rect, results=None, uniq_tree=None):
        if results is None:
            rect = _normalize_rect(rect)
            results = []
            uniq_tree = set()
        # search children
        if self.children:
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[0]._intersect_tree(rect, results, uniq_tree)
                    if rect[7] >= self.center[2]:
                        self.children[1]._intersect_tree(rect, results, uniq_tree)
                if rect[6] >= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[2]._intersect_tree(rect, results, uniq_tree)
                    if rect[7] >= self.center[2]:
                        self.children[3]._intersect_tree(rect, results, uniq_tree)
            if rect[5] >= self.center[0]:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[4]._intersect_tree(rect, results, uniq_tree)
                    if rect[7] >= self.center[2]:
                        self.children[5]._intersect_tree(rect, results, uniq_tree)
                if rect[6] >= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[6]._intersect_tree(rect, results, uniq_tree)
                    if rect[7] >= self.center[2]:
                        self.children[7]._intersect_tree(rect, results, uniq_tree)
        # search node at this level
        for node in self.nodes:
            if node.rect[5] >= rect[0] and node.rect[0] <= rect[5] and \
            node.rect[7] >= rect[1] and node.rect[1] <= rect[7] and \
            node.rect[6] >= rect[2] and node.rect[2] <= rect[6]:
                if id(self) not in uniq_tree:
                    results.append(self)
                    uniq_tree.add(id(self))
                break
        return results


    def _intersect_all_tree(self, rect, results=None, uniq_tree=None):
        hit = False
        if results is None:
            rect = _normalize_rect(rect)
            results = []
            uniq_tree = set()
        # search children
        if self.children:
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        _, hit_res = self.children[0]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
                    if rect[7] >= self.center[2]:
                        _, hit_res = self.children[1]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
                if rect[6] >= self.center[1]:
                    if rect[2] <= self.center[2]:
                        _, hit_res = self.children[2]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
                    if rect[7] >= self.center[2]:
                        _, hit_res = self.children[3]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
            if rect[5] >= self.center[0]:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        _, hit_res = self.children[4]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
                    if rect[7] >= self.center[2]:
                        _, hit_res = self.children[5]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
                if rect[6] >= self.center[1]:
                    if rect[2] <= self.center[2]:
                        _, hit_res = self.children[6]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
                    if rect[7] >= self.center[2]:
                        _, hit_res = self.children[7]._intersect_all_tree(rect, results, uniq_tree)
                        if hit_res:
                            hit = True
                            if id(self) not in uniq_tree:
                                results.append((self._depth, self))
                                uniq_tree.add(id(self))
        # search node at this level
        for node in self.nodes:
            if node.rect[5] >= rect[0] and node.rect[0] <= rect[5] and \
            node.rect[7] >= rect[1] and node.rect[1] <= rect[7] and \
            node.rect[6] >= rect[2] and node.rect[2] <= rect[6]:
                hit = True
                if id(self) not in uniq_tree:
                    results.append((self._depth, self))
                    uniq_tree.add(id(self))
                break
        return results, hit

    def _insert_into_children(self, item, rect):
        # if rect spans center then insert here
        if rect[0] <= self.center[0] and rect[5] >= self.center[0] \
                and rect[1] <= self.center[1] and rect[6] >= self.center[1] \
                and rect[2] <= self.center[2] and rect[7] >= self.center[2]:
            node = _OctNode(item, rect)
            self.nodes.append(node)
        else:
            # try to insert into children
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[0]._insert(item, rect)
                    else:
                        self.children[4]._insert(item, rect)
                else:
                    if rect[2] <= self.center[2]:
                        self.children[2]._insert(item, rect)
                    else:
                        self.children[6]._insert(item, rect)
            else:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[1]._insert(item, rect)
                    else:
                        self.children[5]._insert(item, rect)
                else:
                    if rect[2] <= self.center[2]:
                        self.children[3]._insert(item, rect)
                    else:
                        self.children[7]._insert(item, rect)

    def _remove_from_children(self, item, rect):
        # if rect spans center then insert here
        if rect[0] <= self.center[0] and rect[5] >= self.center[0] \
                and rect[1] <= self.center[1] and rect[6] >= self.center[1] \
                and rect[2] <= self.center[2] and rect[7] >= self.center[2]:
            node = _OctNode(item, rect)
            self.nodes.remove(node)
        else:
            # try to remove from children
            if rect[0] <= self.center[0]:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[0]._remove(item, rect)
                    else:
                        self.children[4]._remove(item, rect)
                else:
                    if rect[2] <= self.center[2]:
                        self.children[2]._remove(item, rect)
                    else:
                        self.children[6]._remove(item, rect)
            else:
                if rect[1] <= self.center[1]:
                    if rect[2] <= self.center[2]:
                        self.children[1]._remove(item, rect)
                    else:
                        self.children[5]._remove(item, rect)
                else:
                    if rect[2] <= self.center[2]:
                        self.children[3]._remove(item, rect)
                    else:
                        self.children[7]._remove(item, rect)

    def _split(self):
        octwidth = self.width / 4.0
        ocheight = self.height / 4.0
        ocdepth = self.depth / 4.0
        halfwidth = self.width / 2.0
        halfheight = self.height / 2.0
        halfdepth = self.depth / 2.0
        x1 = self.center[0] - octwidth
        x2 = self.center[0] + octwidth
        y1 = self.center[1] - ocheight
        y2 = self.center[1] + ocheight
        z1 = self.center[2] - ocdepth
        z2 = self.center[2] + ocdepth
        new_depth = self._depth + 1
        self.children = [
            _OctTree(x1, y1, z1, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
            _OctTree(x1, y2, z1, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
            _OctTree(x2, y1, z1, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
            _OctTree(x2, y2, z1, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
            _OctTree(x1, y1, z2, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
            _OctTree(x1, y2, z2, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
            _OctTree(x2, y1, z2, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
            _OctTree(x2, y2, z2, halfwidth, halfheight, halfdepth, self.max_items, self.max_depth, new_depth),
        ]
        nodes = self.nodes
        self.nodes = []
        for node in nodes:
            self._insert_into_children(node.item, node.rect)

    def __len__(self):
        size = 0
        for child in self.children:
            size += len(child)
        size += len(self.nodes)
        return size
    
        

class OctreeIndex(_OctTree):
    """
    The top spatial index to be created by the user. Once created it can be
    populated with geographically placed members that can later be tested for
    intersection with a user inputted geographic bounding box. Note that the
    index can be iterated through in a for-statement, which loops through all
    all the oct instances and lets you access their properties.

    Example usage:

    >>> spindex = OctreeIndex(bbox=(0, 0, 0, 100, 100, 100))
    >>> spindex.insert('duck', (50, 30, 20, 53, 60, 30))
    >>> spindex.insert('cookie', (10, 20, 30, 15, 25, 40))
    >>> spindex.insert('python', (40, 50, 10, 95, 90, 70))
    >>> results = spindex.intersect((51, 51, 51, 86, 86, 86), method="all_tree")
    >>> sorted(results)
    [(0, <__main__._OctTree object at 0x...>), (1, <__main__._OctTree object at 0x...>), ...]
    """

    def __init__(self, bbox=None, x=None, y=None, z=None, width=None, height=None, depth=None, max_items=None,
                 max_depth=None):
        """
        Initiate by specifying either 1) a bbox to keep track of, or 2) with an xyz centerpoint and a width, height, and depth.

        Parameters:
        - **bbox**: The coordinate system bounding box of the area that the octree should
            keep track of, as a 6-length sequence (xmin, ymin, zmin, xmax, ymax, zmax)
        - **x**:
            The x center coordinate of the area that the octree should keep track of.
        - **y**
            The y center coordinate of the area that the octree should keep track of.
        - **z**
            The z center coordinate of the area that the octree should keep track of.
        - **width**:
            How far from the x center that the octree should look when keeping track.
        - **height**:
            How far from the y center that the octree should look when keeping track.
        - **depth**:
            How far from the z center that the octree should look when keeping track.
        - **max_items** (optional): The maximum number of items allowed per octant before splitting
            up into eight new sub-octants. Default is 10.
        - **max_depth** (optional): The maximum levels of nested sub-octants, after which no more splitting
            occurs and the bottommost oct nodes may grow indefinitely. Default is 20.
        """
        if bbox is not None:
            x1, y1, z1, x2, y2, z2 = bbox
            width, height, depth = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
            midx, midy, midz = x1 + width / 2.0, y1 + height / 2.0, z1 + depth / 2.0
            super(OctreeIndex, self).__init__(midx, midy, midz, width, height, depth, max_items, max_depth)

        elif None not in (x, y, z, width, height, depth):
            super(OctreeIndex, self).__init__(x, y, z, width, height, depth, max_items, max_depth)

        else:
            raise Exception(
                "Either the bbox argument must be set, or the x, y, z, width, height, and depth arguments must be set")

    def insert(self, item, bbox):
        """
        Inserts an item into the octree along with its bounding box.

        Parameters:
        - **item**: The item to insert into the index, which will be returned by the intersection method
        - **bbox**: The spatial bounding box tuple of the item, with six members (xmin, ymin, zmin, xmax, ymax, zmax)
        """
        self._insert(item, bbox)

    def remove(self, item, bbox):
        """
        Removes an item from the octree.

        Parameters:
        - **item**: The item to remove from the index
        - **bbox**: The spatial bounding box tuple of the item, with six members (xmin, ymin, zmin, xmax, ymax, zmax)

        Both parameters need to exactly match the parameters provided to the insert method.
        """
        self._remove(item, bbox)

    def intersect(self, bbox, method):
        """
        Intersects an input bounding box rectangle with all of the items
        contained in the octree.

        Parameters:
        - **bbox**: A spatial bounding box tuple with six members (xmin, ymin, zmin, xmax, ymax, zmax)

        Returns:
        - A list of inserted items whose bounding boxes intersect with the input bbox.
        """
        if method == "node":
            return self._intersect_node(bbox)
        elif method == "tree":
            return self._intersect_tree(bbox)
        elif method == "all_tree":
            res, _ = self._intersect_all_tree(bbox)
            res = sorted(res, key=itemgetter(0), reverse=False)  
            return res
       