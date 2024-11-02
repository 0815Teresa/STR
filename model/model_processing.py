import os

def deleteHistoryModelPath(path):
    file_path = "/".join(path.split("/")[:-1])
    file_name = " ".join(path.split("/")[-1].split(" ")[:-1])
    filelist=os.listdir(file_path)
    for f in filelist:
        if file_name in f:
            del_file = file_path + '/' + f # Absolute paths must be used when the code is not in the same folder as the files to be deleted
            os.remove(del_file)
                         
    
    


    


