import sys
sys.path.append("model")

import copy
import time
import datetime
import pandas as pd
import pickle
import numpy as np

from tqdm import tqdm
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from preprocess.utils import LoadSave
from ocTree.buildTree import build_tree
from ocTree.octree import get_octree_feat
from ocTree.dataLoader import TrajTokenDataLoader

from lossFunc import RankingLoss
from accFunc import topk_acc

from model_processing import deleteHistoryModelPath

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


class ExpConfig(object):
    def __init__(self, config, gpu_id):
        self.config = config
        self.device = self._acquire_device(gpu_id)

    def _acquire_device(self, gpu_id):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Use GPU: cuda {gpu_id}")
        else:
            device = torch.device("cpu")
            print(f"Use CPU")
        
        return device

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) 
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) 
        self.w_2 = nn.Linear(d_hid, d_in) 
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, another_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output1, enc_slf_attn1 = self.slf_attn(
            enc_input, another_input, another_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        enc_output1 = self.pos_ffn(enc_output1)
        return enc_output, enc_output1, enc_slf_attn

    
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach() 
    
class Encoder(nn.Module):
    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200):

        super().__init__()
        self.src_word_emb = nn.Linear(n_src_vocab,d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_seq2, src_mask, return_attns=False):

        enc_slf_attn_list = []

        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)

        enc_output2 = self.dropout(self.position_enc(self.src_word_emb(src_seq2.type_as(src_seq))))
        enc_output2 = self.layer_norm(enc_output2)

        for enc_layer in self.layer_stack:
            enc_output, enc_output2, enc_slf_attn = enc_layer(enc_output, enc_output2, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output+enc_output2,



def get_pad_mask(seq, pad_idx):
    # (batch, seqlen) -> (batch, 1, seqlen) 
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

 
class STRmodel(nn.Module):
    def __init__(
            self, n_src_vocab, 
            d_word_vec=16, d_model=16, d_inner=64,
            n_layers=2, n_head=1, d_k=8, d_v=8, dropout=0.1, n_position=200):

        super().__init__()

        # Encoder
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

    def forward(self, src_seq, src_seq2):
        enc_output, *_ = self.encoder(src_seq, src_seq2, None)

        return torch.mean(enc_output,dim=1)


def pload(file_path):
    with open(file_path, "rb") as tar:
        out = pickle.load(tar)
    return out


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print("MODEL Total parameters:", total_param, "\n")
    return total_param


class ExpSTRmodel(ExpConfig):
    def __init__(self, config, gpu_id, load_model, just_embeddings):
        self.load_model = load_model
        self.store_embeddings = just_embeddings
        self.trajs = pload(config["traj_path"])
        
        super(ExpSTRmodel, self).__init__(config, gpu_id)
        seed_torch(config["seed"])
        
        if just_embeddings:  
            self.octree = build_tree(pload(self.config["traj_path"]), self.config["x_range"], self.config["y_range"], self.config["z_range"], self.config["max_nodes"], self.config["max_depth"])
            self.treeid_range, self.treeid_list_list = get_octree_feat(self.octree, self.config["traj_num"], self.config["traj_size"])
            self.merge_trajs_data = self._merge_data(self.trajs)
            self.nodes_num_all = self._compute_nodes_num()
            self.edgs_adj = self._compute_common_tps(self.nodes_num_all)
            self.embeding_loader = self._get_dataloader(flag="embed")
        else:
            self.log_writer = SummaryWriter(f"./runs/{self.config['data']}/{self.config['length']}/{self.config['model']}_{self.config['dis_type']}_{datetime.datetime.now()}/")

            print("[!] Build octree, max nodes:", self.config["max_nodes"], "max depth:", self.config["max_depth"], "x_range:", self.config["x_range"], "y_range:", self.config["y_range"])
            self.octree = build_tree(pload(self.config["traj_path"])[:self.config["val_data_range"][1]], self.config["x_range"], self.config["y_range"], self.config["z_range"], self.config["max_nodes"], self.config["max_depth"])

            self.treeid_range, self.treeid_list_list = get_octree_feat(self.octree, self.config["traj_num"], self.config["traj_size"])

            self.merge_trajs_data = self._merge_data(self.trajs)

            self.nodes_num_all = self._compute_nodes_num()
            self.edgs_adj = self._compute_common_tps(self.nodes_num_all)

            self.train_loader = self._get_dataloader(flag="train")
            print("Training Graphs: ", len(self.train_loader.dataset))

            self.val_loader = self._get_dataloader(flag="val")
            print("Validation Graphs: ", len(self.val_loader.dataset))

        self.model = self._build_model().to(self.device)

    def _build_model(self):
        if self.config["model"] == "STR":
            model = STRmodel(n_src_vocab=self.config["in_features"], d_word_vec=self.config['d_word_vec'], d_model=self.config['d_model'], d_inner=self.config['d_inner'], n_layers=self.config['n_layers'], n_head=self.config['n_head'], d_k=self.config['d_k'], d_v=self.config['d_v'], dropout=self.config['dropout'], n_position=self.config["traj_size"])

        view_model_param(model)

        if self.load_model is not None:
            ck = torch.load(self.load_model)
            model.load_state_dict(ck,strict=False)
            print("[!] Load model weight:", self.load_model)

        return model

    def _get_dataloader(self, flag):
        if flag == "train":
            trajs = self.merge_trajs_data[self.config["train_data_range"][0] : self.config["train_data_range"][1]]
            print("Train traj number:", len(trajs))
            matrix = pload(self.config["stdis_matrix_path"].format(self.config["dis_type"]))[self.config["train_data_range"][0] : self.config["train_data_range"][1], self.config["train_data_range"][0] : self.config["train_data_range"][1]]
            edgs_adj = self.edgs_adj[self.config["train_data_range"][0] : self.config["train_data_range"][1], self.config["train_data_range"][0] : self.config["train_data_range"][1]]
        elif flag == "val":
            trajs = self.merge_trajs_data
            print("Val traj number:", len(trajs))
            matrix = pload(self.config["stdis_matrix_path"].format(self.config["dis_type"]))[self.config["val_data_range"][0] : self.config["val_data_range"][1], :self.config["val_data_range"][1]]
            edgs_adj = self.edgs_adj
        elif flag == "embed":
            trajs = self.merge_trajs_data
            matrix = pload(self.config["stdis_matrix_path"].format(self.config["dis_type"]))
            edgs_adj = self.edgs_adj

        data_loader = TrajTokenDataLoader(traj_data=trajs, dis_matrix=matrix, edgs_adj=edgs_adj, phase=flag, train_batch_size=self.config["train_batch_size"], eval_batch_size=self.config["eval_batch_size"], sample_num=self.config["sample_num"], num_workers=self.config["num_workers"], data_features=self.config["data_features"], x_range=self.config["x_range"], y_range=self.config["y_range"], z_range=self.config["z_range"], treeid_list_list=self.treeid_list_list, treeid_range=self.treeid_range).get_data_loader()

        return data_loader

    def _merge_data(self, trajs):
        merge_trajs = [[[trajs[i][j][0], trajs[i][j][1], trajs[i][j][2], self.treeid_list_list[i][j][0], self.treeid_list_list[i][j][1], self.treeid_list_list[i][j][2], self.treeid_list_list[i][j][3]] for j in range(len(trajs[i]))] for i in range(len(trajs))]
        return merge_trajs
    
    # Find how many trajectory points are in each grid for each trajectory
    def _compute_nodes_num(self):
        nodes_num_all = []
        for treeid_list in self.treeid_list_list:
            nodes_num = {}
            for treeid in treeid_list:
                if treeid == 0:
                    continue
                else:
                    key = str(treeid[0]) + "+" + str(treeid[1])
                    if key in nodes_num.keys():
                        nodes_num[key] += 1
                    else:
                        nodes_num[key] = 1
            nodes_num_all.append(nodes_num)
        return nodes_num_all

    # Between trajectories and trajectories based on the number of shared octree lattices and the density of points within the lattice, the two-by-two similarity is calculated and normalized between 0-1
    def _compute_common_tps(self, node_num_all):
        edge_adj = [[0 for _ in range(1000)] for _ in range(1000)]
        for i in range(len(node_num_all)):
            traj1 = node_num_all[i]
            for j in range(i, len(node_num_all)):
                traj2 = node_num_all[j]
                tps_num = 0
                for key,value in traj1.items():
                    if key in traj2:
                        tps_num += min(value, traj2[key])
                    else:
                        continue
                if i == j:
                    edge_adj[i][i] = tps_num/len(self.merge_trajs_data[i])*0.5 + tps_num/len(self.merge_trajs_data[j])*0.5
                else:
                    edge_adj[i][j] = tps_num/len(self.merge_trajs_data[i])*0.5 + tps_num/len(self.merge_trajs_data[j])*0.5
                    edge_adj[j][i] = tps_num/len(self.merge_trajs_data[i])*0.5 + tps_num/len(self.merge_trajs_data[j])*0.5
               
        return torch.tensor(edge_adj).double()
    
    def _get_edge_index(self, edgs_adj):
        edge_index = torch.nonzero(edgs_adj).T
        edge_weight = edgs_adj[edge_index[0],edge_index[1]]/10
        return edge_index, edge_weight
    
    def _select_optimizer(self):
        if self.config["optimizer"] == "SGD":
            model_optim = optim.SGD(self.model.parameters(), lr=self.config["init_lr"])
        elif self.config["optimizer"] == "Adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.config["init_lr"])

        return model_optim, None

    def _select_criterion(self):
        criterion = RankingLoss(self.config["sample_num"], self.config["alpha"], self.device).float()

        return criterion

    def embedding(self):
        all_vectors = []
        self.model.eval()

        loader_time = 0
        begin_time = time.time()

        for data in tqdm(self.embeding_loader):
            traj_feature_list_list, dis_list_list, idx, sample_index, sim_trajs = data
            traj_feature_list_list_tensor = [torch.Tensor(item.detach().numpy()).to(self.device) for items in traj_feature_list_list for item in items]

            with torch.set_grad_enabled(True):
                for traj_feature_list,sim_traj in zip(traj_feature_list_list_tensor, sim_trajs):
                    vectors = self.model(traj_feature_list, sim_traj.to(self.device)) 
                    all_vectors.append(vectors)
                    
        all_vectors = torch.stack(all_vectors).squeeze()
        print("all_embeding_vectors length:", len(all_vectors))
        print("all_embedding_vectors shape:", all_vectors.shape)

        end_time = time.time()
        print(f"all embedding time: {end_time-begin_time-loader_time} seconds")

        hr10, hr50, r10_50 = topk_acc(row_embedding_tensor=all_vectors, col_embedding_tensor=all_vectors, distance_matrix=self.embeding_loader.dataset.dis_matrix, matrix_cal_batch=self.config["matrix_cal_batch"],)

        print(hr10, hr50, r10_50)
        file_processor = LoadSave()
        file_processor.save_data(all_vectors, self.config["embeddings_path"])
        
    def val(self):
        all_vectors = []
        self.model.eval()

        for data in self.val_loader:
            traj_feature_list_list, dis_list_list, idx, sample_index, sim_trajs = data
            traj_feature_list_list_tensor = [torch.Tensor(item.detach().numpy()).to(self.device) for items in traj_feature_list_list for item in items]

            with torch.set_grad_enabled(True):
                for traj_feature_list,sim_traj in zip(traj_feature_list_list_tensor, sim_trajs):
                    vectors = self.model(traj_feature_list, sim_traj.to(self.device)) 
                    all_vectors.append(vectors)

        all_vectors = torch.stack(all_vectors).squeeze()
        print("all_val_vectors length:", len(all_vectors))

        hr10, hr50, r10_50 = topk_acc(row_embedding_tensor=all_vectors[self.config["val_data_range"][0] : self.config["val_data_range"][1]], col_embedding_tensor=all_vectors, distance_matrix=self.val_loader.dataset.dis_matrix, matrix_cal_batch=self.config["matrix_cal_batch"])
        
        return hr10, hr50, r10_50


    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_hr10 = 0.0
        time_now = time.time()

        model_optim, scheduler = self._select_optimizer()
        criterion = self._select_criterion()

        traj_vec_np = np.empty([int(self.config['traj_num']), int(self.config['d_model'])], dtype=float)
        self.traj_vec = torch.tensor(traj_vec_np, dtype=float) 

        for epoch in range(self.config["epoch"]):
            print("-------epoch: {} starting------".format( epoch+1))
            self.model.train()

            epoch_begin_time = time.time()
            epoch_loss = 0.0

            dataload_time = 0
            embed_time = 0
            test_time = time.time()

            for data in self.train_loader:
                
                traj_feature_list_list, dis_list_list, idx, sample_index, sim_trajs = data
                dataload_time += time.time() - test_time
                test_time2 = time.time()                
                
                loss=0
                for traj_feature_list,dis_list,idx_in,sample_index_in,sim_traj in zip(traj_feature_list_list,dis_list_list,idx,sample_index,sim_trajs):
                    vectors_all = []
                    traj_feature_list_list_tensor = [torch.Tensor(item.detach().numpy()).to(self.device) for items in traj_feature_list for item in items]
                    sim_traj_tensor = sim_traj.to(self.device)
                    is_cross = True

                    with torch.set_grad_enabled(True):
                        for traj_feature_list_tensor in traj_feature_list_list_tensor:
                            vectors = self.model(traj_feature_list_tensor.unsqueeze(dim=0), sim_traj_tensor).squeeze()
                            vectors_all.append(vectors)

                    test_time = time.time()
                    embed_time += time.time() - test_time2

                    vectors_all = torch.stack(vectors_all).unsqueeze(dim=0)

                    loss += criterion(self.config["sample_num"], vectors_all, torch.tensor(dis_list).unsqueeze(dim=0).to(self.device), self.edgs_adj[idx_in,sample_index_in])

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

            epoch_loss += loss.item()

            print("\nLoad data time:", int(dataload_time), "s")
            print("Train model time:", int(embed_time), "s\n")

            epoch_loss = epoch_loss / len(self.train_loader.dataset)
            self.log_writer.add_scalar(f"TrajRepresentation/Loss", float(epoch_loss), epoch)

            epoch_end_time = time.time()
            print(f"\nEpoch {epoch+1}/{self.config['epoch']}:\nTrain Loss: {epoch_loss:.4f}\tTime: {(epoch_end_time - epoch_begin_time) // 60} m {int((epoch_end_time - epoch_begin_time) % 60)} s")

            val_begin_time = time.time()
            hr10, hr50, r10_50 = self.val()
            val_end_time = time.time()

            self.log_writer.add_scalar(f"TrajRepresentation/HR10", hr10, epoch)
            self.log_writer.add_scalar(f"TrajRepresentation/HR50", hr50, epoch)
            self.log_writer.add_scalar(f"TrajRepresentation/R10@50", r10_50, epoch)

            print(f"Val HR10: {100 * hr10:.4f}%\tHR50: {100 * hr50:.4f}%\tR10@50: {100 * r10_50:.4f}%\tTime: {(val_end_time -val_begin_time) // 60} m {int((val_end_time -val_begin_time) % 60)} s")

            if hr10 > best_hr10:
                best_hr10 = hr10
                best_model_wts = copy.deepcopy(self.model.state_dict())
                print(f"Best HR10: {100*best_hr10:.4f}%")
                print(f"model path: ",self.config["model_best_wts_path"].format(best_hr10))
                
                val_res = pd.DataFrame([[hr10, hr50,r10_50]], columns=["HR10","HR50","R10@50"])
                val_res.to_csv(self.config["model_best_topAcc_path"],index=False)
                
                model_file = self.config["model_best_wts_path"].format(best_hr10)
                deleteHistoryModelPath(model_file)
                torch.save(best_model_wts, model_file)

        time_end = time.time()
        print("\nAll training complete in {:.0f}m {:.0f}s".format((time_end - time_now) // 60, (time_end - time_now) % 60))
        print(f"Best HR10: {100*best_hr10:.4f}%")
        torch.save(best_model_wts, self.config["model_best_wts_path"].format(best_hr10))




