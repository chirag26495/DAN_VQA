import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader_dan import get_loader
from models_singleattn3_dan import VqaModel, SANModel, TripletLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def main(args):

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx

#     model = VqaModel(
#         embed_size=args.embed_size,
#         qst_vocab_size=qst_vocab_size,
#         ans_vocab_size=ans_vocab_size,
#         word_embed_size=args.word_embed_size,
#         num_layers=args.num_layers,
#         hidden_size=args.hidden_size).to(device)

    model = SANModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size).to(device)
    # model = torch.jit.script(SANModel(
    #     embed_size=args.embed_size,
    #     qst_vocab_size=qst_vocab_size,
    #     ans_vocab_size=ans_vocab_size,
    #     word_embed_size=args.word_embed_size,
    #     num_layers=args.num_layers,
    #     hidden_size=args.hidden_size)).to(device)

    criterion = nn.CrossEntropyLoss()
    #### margin value to be decided on the basis of validation data
    margin = 0.2
    criterion2 = TripletLoss(margin)
    # criterion2 = torch.jit.script(TripletLoss(margin))
    #### v_weightage value (for triplet vs classification loss ratio) to be decided on the basis of validation data
    v_weightage = 1.0

    #params = list(model.qst_encoder.parameters()) \
    #    + list(model.san.parameters()) \
    #    + list(model.mlp.parameters())

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    ### Early stopping
    early_stop_threshold = 10 #3
    best_loss = 99999
    val_increase_count = 0
    stop_training = False
    prev_loss = 9999

    for epoch in range(args.num_epochs):

        for phase in ['train', 'valid']:

            running_loss = 0.0
            running_ce_loss = 0.0
            running_triplet_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size

            if phase == 'train':
                # scheduler.step()
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):

                image = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
                
                supporting_example_image = batch_sample['supporting_example_image'].to(device)
                # supporting_example_question = batch_sample['supporting_example_question'].to(device)
                opposing_example_image = batch_sample['opposing_example_image'].to(device)
                # opposing_example_question = batch_sample['opposing_example_question'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    output, attn_scores = model(image, question)      # [batch_size, ans_vocab_size=1000, attn_scores=196]
                    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    ce_loss = criterion(output, label)
                    
                    _, supporting_example_attn_scores = model(supporting_example_image, question)      # [batch_size, ans_vocab_size=1000, attn_scores=196]
                    _, opposing_example_attn_scores = model(opposing_example_image, question)      # [batch_size, ans_vocab_size=1000, attn_scores=196]
                    
                    triplet_loss = criterion2(attn_scores, supporting_example_attn_scores, opposing_example_attn_scores)
                    loss = ce_loss + v_weightage * triplet_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += loss.item()
                running_ce_loss += ce_loss.item()
                running_triplet_loss += triplet_loss.item()
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                if batch_idx % 100 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}, ce_Loss: {:.4f}, triplet_Loss: {:.4f}'
                          .format(phase.upper(), epoch+1, args.num_epochs, batch_idx, int(batch_step_size), loss.item(), ce_loss.item(), triplet_loss.item()))

            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            epoch_ce_loss = running_ce_loss / batch_step_size
            epoch_triplet_loss = running_triplet_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)      # multiple choice

            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, ce_Loss: {:.4f}, triplet_Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
                  .format(phase.upper(), epoch+1, args.num_epochs, epoch_loss, epoch_ce_loss, epoch_triplet_loss, epoch_acc_exp1, epoch_acc_exp2))

            # Log the loss and accuracy in an epoch.
            with open(os.path.join(args.log_dir, '{}-{}-log-epoch-{:02}.txt')
                      .format(args.model_name, phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc_exp1.item()) + '\t'
                        + str(epoch_acc_exp2.item()) + '\t'
                        + str(epoch_ce_loss) + '\t'
                        + str(epoch_triplet_loss))

            if phase == 'valid':
                if epoch_loss < best_loss:
                    print("At epoch:",epoch+1,"best loss from:\t",best_loss, "\tto\t",epoch_loss)
                    best_loss = epoch_loss
                    torch.save(model, os.path.join(args.model_dir, '{}-best_model.pt'.format(args.model_name)))
                    torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()}, os.path.join(args.model_dir, '{}-best_model.ckpt'.format(args.model_name)))
                if epoch_loss > prev_loss:
                    val_increase_count += 1
                else:
                    val_increase_count = 0
                if val_increase_count >= early_stop_threshold:
                    stop_training = True
                prev_loss = epoch_loss
        # # Save the model check points.
        # if (epoch+1) % args.save_step == 0:
        #     torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
        #                os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))
        
        scheduler.step()
        print("lr val:",optimizer.state_dict()['param_groups'][0]['lr'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        help='model name.') 

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default='./models',
                        help='directory for saved models.')

    parser.add_argument('--max_qst_length', type=int, default=30,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=10,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--hidden_size', type=int, default=512,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=10,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=256,#2,#256, #224, #128, #256,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()

    main(args)
