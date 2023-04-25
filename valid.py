import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from data_loader import get_loader
from data_loader_dan_3 import get_loader

# from models_5 import VqaModel
# from models_singleattn import SANModel
# from models_4 import VqaModel
from models_singleattn3_dan import SANModel
# from dan_best_neeraj import SANModel

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

phase = 'valid'
input_dir = './datasets'
max_qst_length = 30
max_num_ans = 10
batch_size = 256
num_workers = 8

ans_type=['', '_bool', '_number', '_other']
idx = 0


data_loader = get_loader(
        input_dir=input_dir,
        input_vqa_train=f'train{ans_type[idx]}.npy',
        input_vqa_valid=f'valid{ans_type[idx]}.npy',
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers)

qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx
print(qst_vocab_size, ans_vocab_size, ans_unk_idx)

embed_size = 1024
word_embed_size = 300
num_layers = 2
hidden_size = 512
triplet_loss = 0

model = VqaModel(
# model = SANModel(
    embed_size=embed_size,
    qst_vocab_size=qst_vocab_size,
    ans_vocab_size=ans_vocab_size,
    word_embed_size=word_embed_size,
    num_layers=num_layers,
    hidden_size=hidden_size)

if torch.cuda.device_count() > 0:
    print("Using", torch.cuda.device_count(), "GPUs.")
    # dim = 0 [40, xxx] -> [10, ...], [10, ...], [10, ...], [10, ...] on 4 GPUs
    model = nn.DataParallel(model)

model = model.to(device)

# model.load_state_dict(torch.load('./models/basic_2-epoch-30.ckpt')['state_dict'])
# model.load_state_dict(torch.load('./models/chirag/san_1stack-best_model.ckpt')['state_dict'])
# model.load_state_dict(torch.load('./models/san_2-epoch-30.ckpt')['state_dict'])
model.load_state_dict(torch.load('./models/dan_pi_farther_neighbours_1-best_model.ckpt')['state_dict'])
# model.load_state_dict(torch.load('./models/chirag/dan-best_model.ckpt')['state_dict'])

model.eval()


running_corr_exp1 = 0
running_corr_exp2 = 0

batch_step_size = len(data_loader[phase].dataset) / batch_size

# for batch_idx, batch_sample in enumerate(tqdm(data_loader['train'])):
for batch_idx, batch_sample in enumerate(tqdm(data_loader['valid'])):
    image_index = batch_sample['index']
    
    image = batch_sample['image'].to(device)
    question = batch_sample['question'].to(device)
    label = batch_sample['answer_label'].to(device)
    multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

    # output = model(image, question)      # [batch_size, ans_vocab_size=1000]
    output, attn_scores = model(image, question)      # [batch_size, ans_vocab_size=1000]

    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
    _, pred_exp2 = torch.max(output, 1)  # [batch_size]
    
    # Evaluation metric of 'multiple choice'
    # Exp1: our model prediction to '<unk>' IS accepted as the answer.
    # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
    pred_exp2[pred_exp2 == ans_unk_idx] = -9999
    running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
    running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()


# Print the average loss and accuracy in an epoch.
epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      # multiple choice
epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)      # multiple choice

print('| {} SET | Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
      .format(phase.upper(), epoch_acc_exp1, epoch_acc_exp2), flush=True)
