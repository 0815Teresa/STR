import torch
import numpy as np

def topk_acc(row_embedding_tensor, col_embedding_tensor, distance_matrix, matrix_cal_batch):
    row_embedding_num = len(row_embedding_tensor)  
    col_embedding_num = len(col_embedding_tensor) 
    print(f"val matrix shape: ({row_embedding_num},{col_embedding_num})")
    hr_10, hr_50, r10_50 = 0.0, 0.0, 0.0
    truth_dis_matrix = distance_matrix  
    predic_dis_matrix = np.zeros((row_embedding_num, col_embedding_num))

    embeddings_index = 0

    while embeddings_index < row_embedding_num:
        cal_batch_row = []
        cal_batch_col = []
        if embeddings_index + matrix_cal_batch <= row_embedding_num:
            end_index = embeddings_index + matrix_cal_batch
        else:
            end_index = row_embedding_num

        for this_index in range(embeddings_index, end_index):
            this_vec = row_embedding_tensor[this_index]
            this_vec = this_vec.unsqueeze(0)
            cal_batch_row.append(this_vec.repeat(col_embedding_num, 1))
            cal_batch_col.append(col_embedding_tensor)

        cal_batch_row = torch.cat(cal_batch_row, dim=0)
        cal_batch_col = torch.cat(cal_batch_col, dim=0)

        batch_out = torch.norm(cal_batch_row - cal_batch_col, p=2, dim=1)

        for batch_out_index, row_index in enumerate(range(embeddings_index, end_index)):
            predic_dis_matrix[row_index] = batch_out[batch_out_index * col_embedding_num : (batch_out_index + 1) * col_embedding_num].cpu().detach().numpy()


        embeddings_index = end_index

    # print(predic_dis_matrix)
    for i in range(row_embedding_num):
        truth_sort_index = np.argsort(truth_dis_matrix[i])
        predic_sort_index = np.argsort(predic_dis_matrix[i])

        hr_10 += len(set(truth_sort_index[:11]) & set(predic_sort_index[:11])) - 1
        hr_50 += len(set(truth_sort_index[:51]) & set(predic_sort_index[:51])) - 1
        r10_50 += len(set(truth_sort_index[:11]) & set(predic_sort_index[:51])) - 1

    return (
        hr_10 / row_embedding_num / 10,   # HR@10
        hr_50 / row_embedding_num / 50,   # HR@50
        r10_50 / row_embedding_num / 10,   # R10@50
    )

