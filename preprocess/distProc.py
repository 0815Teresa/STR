import pickle as pickle
import numpy as np
from distFunc import trajectory_distance_combain, trajecotry_distance_list, trajecotry_temporal_distance_list, trajectory_temporal_distance_combain, trajectory_spatial_temporal_simility

def distance_comp(coor_path, data_name, trajs_len, batch_size, distance_type="discret_frechet", save_path = "./data/features/"):
    traj_coord = pickle.load(open(coor_path, 'rb'))
    np_traj_coord = []
    for t in traj_coord:
        np_traj_coord.append(np.array(t))
    print("the length of np_traj_coord",len(np_traj_coord))

    trajecotry_distance_list(np_traj_coord, batch_size=batch_size, processors=15, distance_type=distance_type,
                             data_name=data_name,save_path = save_path)
    all_spatial_dis = trajectory_distance_combain(trajs_len, batch_size=batch_size, metric_type=distance_type, data_name=data_name,save_path = save_path)
    
    trajecotry_temporal_distance_list(np_traj_coord, batch_size=batch_size, processors=15, 
                             data_name=data_name,save_path = save_path)
    all_temporal_dis = trajectory_temporal_distance_combain(trajs_len, batch_size=batch_size, data_name=data_name,save_path = save_path)
    
    spatial_temporal_dis = trajectory_spatial_temporal_simility(all_spatial_dis,all_temporal_dis)
    
    pickle.dump(spatial_temporal_dis, open(save_path + data_name.split(".")[0] + '_' + distance_type + '_st_distance_all_' + str(trajs_len) + '.pkl', 'wb'))
    print(111)
    



