import pandas 
import argparse
import yaml
import warnings
import torch

from model.STRmodel import ExpSTRmodel

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STR")
    parser.add_argument("-C", "--config", type=str)
    parser.add_argument("-D", "--dis_type", type=str, default="dtw")
    parser.add_argument("-T", "--traj_num", type=str, default=1000)
    parser.add_argument("-X", "--data", type=str, default="porto")
    parser.add_argument("-G", "--gpu", type=str, default="0")
    parser.add_argument("-L", "--load-model", type=str)
    parser.add_argument("-J", "--just_embedding", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config["dis_type"] = args.dis_type
    config["traj_num"] = args.traj_num
    config["data"] = config["data"].format(args.data)
    config["traj_path"] = config["traj_path"].format(config["data"])
    
    config["stdis_matrix_path"] =config["stdis_matrix_path"].format(args.data, config["dis_type"], config["traj_num"])
    
    config["model_best_wts_path"] = config["model_best_wts_path"].format(config["data"], config["length"], config["model"], config["dis_type"]) + " {:.4f}.pkl"
    config["model_best_topAcc_path"] =  config["model_best_topAcc_path"].format(config["data"], config["length"], config["model"], config["dis_type"])
    
    config["embeddings_path"] =  config["embeddings_path"].format(config["data"], config["length"], config["model"], config["dis_type"])

    print("Args in experiment:")
    print(config)
    print("GPU:", args.gpu)
    print("Load model:", args.load_model)
    print("Store embeddings:", args.just_embedding, "\n")

    if args.just_embedding:
        ExpSTRmodel(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).embedding()
    else:
        ExpSTRmodel(config=config, gpu_id=args.gpu, load_model=args.load_model, just_embeddings=args.just_embedding).train()

    torch.cuda.empty_cache()
