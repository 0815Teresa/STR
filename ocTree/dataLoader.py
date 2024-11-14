import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrajDataset(Dataset):
    def __init__(self, traj_data, dis_matrix, edgs_adj, phase, sample_num):
        self.traj_data = traj_data  # [(), (), ()]
        self.dis_matrix = dis_matrix
        self.phase = phase
        self.sample_num = sample_num
        self.edgs_adj = edgs_adj

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.traj_data)

    def __getitem__(self, idx):
        traj_list = []
        dis_list = []

        top_indices = np.argsort(-self.edgs_adj[idx])[1:11]  
        id_most_sim = np.random.choice(top_indices)  
        sim_traj = self.traj_data[id_most_sim]

        if self.phase == "train":
            id_list = np.argsort(self.dis_matrix[idx])

            sample_index = []
            sample_index.extend(id_list[: self.sample_num // 2])  
            sample_index.extend(id_list[len(id_list) - self.sample_num // 2 :]) 

            for i in sample_index:
                traj_list.append(self.traj_data[i])
                dis_list.append(self.dis_matrix[sample_index[0], i])

        elif self.phase == "val" or "test":
            traj_list.append(self.traj_data[idx])
            dis_list = None
            sample_index = None

        return traj_list, dis_list, idx, sample_index, sim_traj


class TrajTokenDataLoader:
    def __init__(self, traj_data, dis_matrix, edgs_adj, phase, train_batch_size, eval_batch_size, sample_num, data_features, num_workers, x_range, y_range, z_range, treeid_list_list, treeid_range):
        self.traj_data = traj_data
        self.dis_matrix = dis_matrix / dis_matrix.max()
        self.edgs_adj = edgs_adj
        self.phase = "val" if phase in ["val", "embed"] else phase
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.sample_num = sample_num
        self.data_features = data_features
        self.num_workers = num_workers
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.treeid_list_list = treeid_list_list
        self.treeid_range = treeid_range

    def get_data_loader(self):
        self.dataset = TrajDataset(traj_data=self.traj_data, dis_matrix=self.dis_matrix, edgs_adj=self.edgs_adj, phase=self.phase, sample_num=self.sample_num)

        if self.phase == "train":
            is_shuffle = False
            batch_size = self.train_batch_size
        elif self.phase == "val" or "test":
            is_shuffle = False
            batch_size = self.eval_batch_size

        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=self.num_workers, collate_fn=self._collate_func)

        return data_loader

    def _collate_func(self, samples):
        traj_list_list, dis_list_list, idx, sample_index, sim_traj = map(list, zip(*samples))
        traj_feature_list_list = self._prepare(traj_list_list)
        sim_traj_norm = self._prepare(sim_traj)
        return traj_feature_list_list, dis_list_list, idx, sample_index, sim_traj_norm

    def _prepare(self, traj_l_l):
        traj_feature_list_list = []
        for traj_l in traj_l_l:
            traj_feature_list = [self._normalize(traj) for traj in traj_l]
            traj_feature_list_list.append(traj_feature_list)
        return traj_feature_list_list

    def _normalize(self, traj):
        # min-max normalization
        traj = torch.tensor(traj)
        lon_min = self.x_range[0]
        lon_max = self.x_range[1]
        lat_min = self.y_range[0]
        lat_max = self.y_range[1]
        t_min = self.z_range[0]
        t_max = self.z_range[1]
        tree_depth_min = self.treeid_range[0]
        tree_depth_max = self.treeid_range[1]
        tree_id_min = self.treeid_range[2]
        tree_id_max = self.treeid_range[3]
        tree_parent_depth_min = self.treeid_range[4]
        tree_parent_depth_max = self.treeid_range[5]
        tree_parent_id_min = self.treeid_range[6]
        tree_parent_id_max = self.treeid_range[7]
        traj = traj - torch.tensor([lon_min, lat_min, t_min, tree_depth_min, tree_id_min, tree_parent_depth_min, tree_parent_id_min])
        traj = traj / torch.tensor([lon_max-lon_min, lat_max-lat_min, t_max-t_min, tree_depth_max-tree_depth_min, tree_id_max-tree_id_min, tree_parent_depth_max-tree_parent_depth_min, tree_parent_id_max-tree_parent_id_min])
        return traj


