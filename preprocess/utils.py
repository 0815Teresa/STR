import os
import pickle
import time
import pandas as pd
import functools
import ast

def timefn(fcn):
    """Decorator for efficency analysis. """
    @functools.wraps(fcn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fcn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fcn.__name__ + " took {:.5f}".format(end-start)
            + " seconds.")
        return result
    return measure_time

class ExpTrajDataPreprocessor(object): 
    
    def __init__(self, withTime):
        self.withTime = withTime
                
    # Stores two formats, no time required for comparison algorithms
    def save(self,data, outpath):
        file_processor = LoadSave()
        if self.withTime:
            file_processor.save_data(data, "../data/processed_data/" +outpath)
        else:
            file_processor.save_data(data, "../data/processed_data/Spa" +outpath)
    
    def add_timestamp_to_polyline(self, row):
        polyline = ast.literal_eval(row['POLYLINE'])
        timestamp = row['TIMESTAMP']
        polyline_with_timestamp = []
        current_time = timestamp
        
        for point in polyline:
            point.append(current_time)
            polyline_with_timestamp.append(point)
            current_time += 15
        
        return polyline_with_timestamp

    def swap_columns(self, data):
        for sublist in data:
            for i in range(len(sublist)):
                sublist[i][0], sublist[i][1] = sublist[i][1], sublist[i][0]
        return data
            
    # input_path: Original trajectory folder path; output_path：Trajectory Save Path
    def _PortoTrajs(self, input_path, output_path):     
        trajs = []
        try:
            df = pd.read_csv(input_path, error_bad_lines=False, warn_bad_lines=True, engine='python')
            df = df[['TIMESTAMP', 'POLYLINE']]
            df['DATE'] = pd.to_datetime(df['TIMESTAMP'], unit='s')
            df['POLYLINE_WITH_TIMESTAMP'] = df.apply(self.add_timestamp_to_polyline, axis=1)
            if self.withTime:
                # (lat,lon,timestamp)
                traj_df = df['POLYLINE_WITH_TIMESTAMP']
                trajs_list = traj_df.tolist()
                result = self.swap_columns(trajs_list)
                for traj in result:
                    small_traj = [traj[i:i+200] for i in range(0,len(traj),200)]
                    trajs.extend(small_traj)
            else:
                # (lat,lon)
                traj_df = df['POLYLINE']
                trajs_list = traj_df.tolist()
                trajs_lists = []
                for traj_df in trajs_list:
                    traj = ast.literal_eval(traj_df)
                    trajs_lists.append(traj)
                result = self.swap_columns(trajs_lists)
                for traj in result:
                    small_traj = [traj[i:i+200] for i in range(0,len(traj),200)]
                    trajs.extend(small_traj)
        except Exception as e:
            print(": _PortoTaxiTrajs: ",e)  
        new_traj_data = [traj for traj in trajs if len(traj) >= 20]
        new_traj_data = new_traj_data[:1500000]
        print("PortoTrajs length: ",len(new_traj_data))
        self.save(new_traj_data,output_path)
        
    def _SanFranciscoTrajs(self, traj_path, output_path):
        fp = open(traj_path+'/cabspottingdata/_cabs.txt', 'r')
        lines = fp.readlines()
        id_list = [line.split("\"")[1] for line in lines]
        raw_df = pd.DataFrame()
        s = 1
        for id in id_list:
            df = pd.read_csv(f"data/raw_data/sanFrancisco/cabspottingdata/new_{id}.txt", header=None, sep=" ")
            df.columns = ['latitude', 'longitude', 'occupancy', 't']
            df.pop('occupancy') 
            df.insert(0, 'id', [id for _ in range(df.shape[0])])  
            raw_df = pd.concat([raw_df, df], axis=0) 
            print('Finished merging {}/{}'.format(s, len(id_list)))
            s += 1
        raw_df = raw_df.sort_values(by=['id', 't'], ascending=[True, True])
        raw_df['date'] = pd.to_datetime(raw_df['t'], unit='s').dt.date
        raw_df['timeStamp'] = pd.to_datetime(raw_df['t'], unit='s')  
        raw_df = raw_df.dropna(axis=0, how='any')
        trajs = []
        try:
            for vehicle_id, group_data in raw_df.groupby(['id','date']):
                if self.withTime:
                    # (lat,lon,timestamp)
                    tmp = group_data[["latitude","longitude","t"]].values
                    small_traj = [tmp[i:i+200] for i in range(0,len(tmp),200)]
                    trajs.extend(small_traj)
                else:
                    # (lat,lon)
                    tmp = group_data[["latitude","longitude"]].values
                    small_traj = [tmp[i:i+200] for i in range(0,len(tmp),200)]
                    trajs.extend(small_traj)   
        except Exception as e:
            print(": _SanFranciscoTrajs: ",e)  
        new_traj_data = [traj for traj in trajs if len(traj) >= 20]
        new_traj_data = new_traj_data[:60000]
        print("SanFranciscoTrajs length: ",len(new_traj_data))
        self.save(new_traj_data,output_path)

    def _RomeTaxi(self, traj_path, output_path):
        trajs = []
        df = pd.read_csv(traj_path, sep=';', header=None, names=['ID', 'timeStamp', 'point'])
        df['timeStamp'] = pd.to_datetime(df['timeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
        df['date'] = df['timeStamp'].dt.date
        df['timeStamp'] = pd.to_datetime(df['timeStamp']).astype('int64')//1e9
        df[['lat', 'lon']] = df['point'].str.extract(r'POINT\(([\d.]+)\s+([\d.]+)\)', expand=True).astype(float)
        df.drop(columns=['point'], inplace=True)
        df = df.dropna(axis=0, how='any')
        trajs = []
        for vehicle_id, group_data in df.groupby(['ID','date']):
            if self.withTime:
                # (lat,lon,timestamp)
                tmp = group_data[["lat","lon","timeStamp"]].values
                small_traj = [tmp[i:i+200] for i in range(0,len(tmp),200)]
                trajs.extend(small_traj)
            else:
                # (lat,lon)
                tmp = group_data[["lat","lon"]].values
                small_traj = [tmp[i:i+200] for i in range(0,len(tmp),200)]
                trajs.extend(small_traj)
        new_traj_data = [traj for traj in trajs if len(traj) >= 10]
        new_traj_data = new_traj_data[:110000]
        print("RomeTrajs length: ",len(new_traj_data))
        self.save(new_traj_data, output_path)
    
class LoadSave():
    """Class for loading and saving object in .pkl format."""
    def __init__(self, file_name=None):
        self._file_name = file_name

    def save_data(self, data=None, path=None):
        """Save data to path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        self.__save_data(data)

    def load_data(self, path=None):
        """Load data from path."""
        if path is None:
            assert self._file_name is not None, "Invaild file path !"
        else:
            self._file_name = path
        return self.__load_data()

    def __save_data(self, data=None):
        """Save data to path."""
        print("--------------Start saving--------------")
        print("@SAVING DATA TO {}.".format(self._file_name))
        with open(self._file_name, 'wb') as file:
            pickle.dump(data, file)
        print("@SAVING SUCESSED !")
        print("----------------------------------------\n")

    def __load_data(self):
        """Load data from path."""
        if not self._file_name:
            raise ValueError("Invaild file path !")
        print("--------------Start loading--------------")
        print("@LOADING DATA FROM {}.".format(self._file_name))
        with open(self._file_name, 'rb') as file:
            data = pickle.load(file)
        print("@LOADING SUCCESSED !")
        print("-----------------------------------------\n")
        return data
    

# View trajectory latitude, longitude, and timestamp ranges
def basic_lat_lon_report(trajs):
    #(lat,lon,t)
    df = []
    for traj in trajs:
        traj_df = pd.DataFrame(traj)
        df.append(traj_df)
    df = pd.concat(df)
    print("lat range: [{} , {}] ".format(df[0].min(), df[0].max()))    
    print("lon range: [{} , {}] ".format(df[1].min(), df[1].max()))  
    print("t range: [{}, {}]".format(df[2].min(), df[2].max()))
    print("lan_mean, lan_std, lot_mean, lot_std, t_mean, t_std: [{},{},{},{},{},{}]".format(df[0].mean(), df[0].std(),df[1].mean(), df[1].std(), df[2].mean(), df[2].std())) 
