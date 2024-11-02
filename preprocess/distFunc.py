import multiprocessing
import pickle

import numpy as np
import traj_dist.distance as tdist

def trajectory_spatial_temporal_simility(spatial_distance, temporal_distance):
    spatial_simility = (spatial_distance - spatial_distance.min())/(spatial_distance.max() - spatial_distance.min())
    temporal_simility = 1 - temporal_distance
    return 0.5*spatial_simility + 0.5*temporal_simility


"""******spatial similarity******"""
def trajecotry_distance_list(trajs, distance_type="hausdorff", batch_size=50, processors=30, data_name='porto', save_path = "./data/features/"):
    pool = multiprocessing.Pool(processes=processors)
    batch_number = 0  
    for i in range(len(trajs)+1):
        if (i != 0) & (i % batch_size == 0):
            print("from {} to {}".format(batch_size * batch_number,i))
            pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size * batch_number:i], trajs, distance_type,
                                                         data_name, save_path))
            batch_number += 1
    pool.close()
    pool.join()


def trajectory_distance_batch(i, batch_trjs, trjs, metric_type="hausdorff", data_name='porto', save_path = "./data/features/") -> None:
    if metric_type == 'lcss' or metric_type == 'edr':
        try:
            trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
        except Exception as e:
            print("tdist.cdist first",e) 
    else:
        try:

            trs_matrix = tdist.cdist(list(batch_trjs), list(trjs), metric=metric_type)
        except Exception as e:
            print("tdist.cdist two",e) 
    print("Shape of trs_matrix: ",trs_matrix.shape)
    pickle.dump(trs_matrix, open(save_path + "tmp/"+ data_name.split(".")[0] + '_' + metric_type + '_distance_' + str(i), 'wb'))

    print('complete: ' + str(i),"\n")


def trajectory_distance_combain(trajs_len, batch_size=100, metric_type="hausdorff", data_name='porto', save_path = "./data/features/"):
    distance_list = []
    for i in range(0, trajs_len + 1):
        if (i != 0) & (i % batch_size == 0):
            temp = pickle.load(open(save_path + "tmp/" + data_name.split(".")[0] + '_' + metric_type + '_distance_' + str(i), "rb"))
            distance_list.append(temp)
            print("Load batches of distance shape: ",distance_list[-1].shape)
    a = distance_list[-1].shape[1]  
    distances = np.array(distance_list) 
    print("Distances shape: ",distances.shape) 
    all_dis = distances.reshape((distances.shape[0]*distances.shape[1],distances.shape[2]))  
    print("After reducing the dimensions and integrating the matrix, Distances shape: " , all_dis.shape)  
    
    pickle.dump(all_dis, open(save_path + data_name.split(".")[0] + '_' + metric_type + '_distance_all_' + str(trajs_len), 'wb'))
    return all_dis


"""******temporal similarity******"""
def trajecotry_temporal_distance_list(trajs, batch_size=50, processors=30, data_name='porto', save_path = "./data/features/"):
    pool = multiprocessing.Pool(processes=processors)
    batch_number = 0  
    for i in range(len(trajs)+1):
        if (i != 0) & (i % batch_size == 0):
            print("from {} to {}".format(batch_size * batch_number,i))
            pool.apply_async(trajectory_temporal_distance_batch
                             ,(i, trajs[batch_size * batch_number:i]
                             ,trajs, data_name, save_path))
            batch_number += 1
    pool.close()
    pool.join()


class TemporalDistance(object):
    def traj_start_end_point(self,traj):
        try:
            start_time = int(traj[0][2])
            end_time = int(traj[-1][2])
            time_diff = end_time - start_time
            print(time_diff)
            return start_time, end_time, time_diff
        except Exception as e:
            print("traj_start_end_point: ", e)  
        
    def _temporal_dist(self,traj_1,traj_2):      
        traj_1_start, traj_1_end, traj_1_diff  = self.traj_start_end_point(traj_1)
        traj_2_start, traj_2_end, traj_2_diff  = self.traj_start_end_point(traj_2)
        diff = max(min(traj_1_end,traj_2_end) - max(traj_1_start,traj_2_start),0)
        return 0.5*(diff/traj_1_diff + diff/traj_2_diff)
    
    def temporal_dist_batch(self,traj_1_list, traj_2_list):
        dis_matrix = np.zeros((len(traj_1_list),len(traj_2_list)))
        for i, traj_1 in enumerate(traj_1_list):
            for j,traj_2 in enumerate(traj_2_list):

                dis_matrix[i][j] = self._temporal_dist(traj_1,traj_2)
        return dis_matrix
    
def trajectory_temporal_distance_batch(i, batch_trjs, trjs, data_name='porto', save_path = "./data/features/"):
    try:
        temporalDistance = TemporalDistance()
        trs_matrix = temporalDistance.temporal_dist_batch(list(batch_trjs), list(trjs))
    except Exception as e:
        print("temporal_dist_batch two",e)  
        
    print("Shape of trs_matrix: ",trs_matrix.shape)
    pickle.dump(trs_matrix, open(save_path + "tmp/" + data_name.split(".")[0] + '_temporal_distance_' + str(i), 'wb'))

    print('complete: ' + str(i),"\n")
    
    
def trajectory_temporal_distance_combain(trajs_len, batch_size=100, data_name='porto', save_path = "./data/features/"):
    distance_list = []
    for i in range(0, trajs_len + 1):
        if (i != 0) & (i % batch_size == 0):
            temp = pickle.load(open(save_path + "tmp/" + data_name.split(".")[0] + '_temporal_distance_' + str(i), "rb"))
            distance_list.append(temp)
            print("Load batched of distance shape: ",distance_list[-1].shape)
    a = distance_list[-1].shape[1]  
    distances = np.array(distance_list) 
    print("Distances shape: ",distances.shape) 
    all_dis = distances.reshape((distances.shape[0]*distances.shape[1],distances.shape[2]))  
    print("After reducing the dimensions and integrating the matrix, Distances shape: " , all_dis.shape)  
    pickle.dump(all_dis, open(save_path + data_name.split(".")[0] + '_temporal_distance_all_' + str(trajs_len), 'wb'))
    return all_dis