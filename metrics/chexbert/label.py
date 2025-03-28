import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from . import utils
from .models.bert_labeler import bert_labeler
from collections import OrderedDict
from .datasets_chexbert.unlabeled_dataset import UnlabeledDataset
from .constants import *
from tqdm import tqdm
import ipdb
def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(tensor_list,
                                                  batch_first=True,
                                                  padding_value=PAD_IDX)
    len_list = [s['len'] for s in sample_list]
    imp_str_list = [s['imp_str'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list, 'imp_str': imp_str_list}
    return batch

def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param csv_path (string): path to csv file containing reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, collate_fn=collate_fn)
    return loader
    
def disable_hooks(model):
    # 清除所有钩子
    model._backward_hooks = {}
    model._backward_pre_hooks = {}
    model._forward_hooks = {}
    model._forward_pre_hooks = {}
    
    # 对所有子模块递归应用
    for child in model.children():
        disable_hooks(child)
    
    return model

def label(model, csv_path, return_probabilities=False):
    """Labels a dataset of reports
    @param model (nn.Module): instantiated CheXbert model
    @param csv_path (string): location of csv with reports

    @returns y_pred (List[List[int]]): Labels for each of the 14 conditions, per report  
    """
    ld = load_unlabeled_data(csv_path, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = disable_hooks(model)
    y_pred = [[] for _ in range(len(CONDITIONS))]
    y_prob = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % 4)
    
    try:
        with torch.no_grad():
            ipdb.set_trace()
            for i, data in enumerate((ld)):
                
                batch = data['imp']  # (batch_size, max_len)
                batch = batch.to(device, dtype=torch.long)
                src_len = data['len']
                attn_mask = utils.generate_attention_masks(batch, src_len, device)
                attn_mask = attn_mask.to(device, dtype=torch.float)

                # print(f"批次形状: {batch.shape}, 注意力掩码形状: {attn_mask.shape}")
                # print(f"批次设备: {batch.device}, 注意力掩码设备: {attn_mask.device}")

                if torch.isnan(batch).any() or torch.isnan(attn_mask).any():
                    print("警告：输入数据中存在NaN值")

                if batch.dim() != 2:
                    print(f"警告：输入批次维度不正确，期望为2，实际为{batch.dim()}")
                
                if batch.shape != attn_mask.shape:
                    print(f"警告：批次形状{batch.shape}与注意力掩码形状{attn_mask.shape}不匹配")

                # 在调用模型前添加安全检查
                try:
                    # 确保输入数据正确
                    if not torch.is_tensor(batch) or not torch.is_tensor(attn_mask):
                        raise TypeError("输入必须是张量")
                    
                    # 检查设备一致性
                    if batch.device != attn_mask.device:
                        attn_mask = attn_mask.to(batch.device)
                    
                    # 使用torch.no_grad确保不创建计算图
                    # ipdb.set_trace()# 预测结果
                    out = model(batch, attn_mask)
                    
                except RuntimeError as e:
                    print(f"模型调用错误: {e}")
                    # 清理缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise

                if len(out) != 14:
                    print(f"警告：模型输出长度不正确，期望为14，实际为{len(out)}")
                for j in range(len(out)):
                    curr_y_pred = out[j].argmax(dim=1)  # shape is (batch_size)
                    curr_y_prob = nn.functional.softmax(out[j], dim=1)[:, 1]  # "positive" probability
                    y_pred[j].append(curr_y_pred.cpu())
                    y_prob[j].append(curr_y_prob.cpu())
                # ipdb.set_trace()
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            for j in range(len(y_pred)):
                y_pred[j] = torch.cat(y_pred[j], dim=0)
                y_prob[j] = torch.cat(y_prob[j], dim=0)
    except Exception as e:
        print(f"模型前向传播错误: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    y_pred = [t.tolist() for t in y_pred]
    y_prob = [t.tolist() for t in y_prob]

    if return_probabilities:
        return y_pred, y_prob
    else:
        return y_pred

def save_preds(y_pred, csv_path, out_path):
    """Save predictions as out_path/labeled_reports.csv 
    @param y_pred (List[List[int]]): list of predictions for each report
    @param csv_path (string): path to csv containing reports
    @param out_path (string): path to output directory
    """
    y_pred = np.array(y_pred)
    y_pred = y_pred.T
    
    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    reports = pd.read_csv(csv_path)['Report Impression']

    df['Report Impression'] = reports.tolist()
    new_cols = ['Report Impression'] + CONDITIONS
    df = df[new_cols]

    df.replace(0, np.nan, inplace=True) #blank class is NaN
    df.replace(3, -1, inplace=True)     #uncertain class is -1
    df.replace(2, 0, inplace=True)      #negative class is 0 
    
    df.to_csv(os.path.join(out_path, 'labeled_reports.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint

    y_pred = label(checkpoint_path, csv_path)
    save_preds(y_pred, csv_path, out_path)
