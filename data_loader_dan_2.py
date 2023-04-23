import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import text_helper


class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, phase='train', transform=None):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa, allow_pickle=True)
        #self.vqa_neighbors = np.load('./neighbor_matrix.npy', allow_pickle=True)
        self.vqa_neighbors = np.load('./neighbor_matrix_small.npy', allow_pickle=True)
        # self.vqa_idx_mapping = np.load('./train_vqa_features.npy', allow_pickle=True).tolist()['vqa_idx']
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform
        self.phase = phase

    def __getitem__(self, idx):

        vqa = self.vqa
        vqa_neighbors = self.vqa_neighbors
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans

        image = vqa[idx]['image_path'].replace('/scratch/cp_wks/codes/basic_vqa','/home2/neeraj.veerla/vqaCV/basic_vqa')
#         print(image)
        image = Image.open(image).convert('RGB')
        
        if(self.phase=='valid'):
            near_idx = np.random.randint(1,49,1)[0]
            s_image = vqa[vqa_neighbors[idx, near_idx]]['image_path'].replace('/scratch/cp_wks/codes/basic_vqa','/home2/neeraj.veerla/vqaCV/basic_vqa')
            s_image = Image.open(s_image).convert('RGB')
        
            #far_idx = np.random.randint(1100,1200,1)[0]
            far_idx = np.random.randint(51,149,1)[0]
            o_image = vqa[vqa_neighbors[idx, far_idx]]['image_path'].replace('/scratch/cp_wks/codes/basic_vqa','/home2/neeraj.veerla/vqaCV/basic_vqa')
            o_image = Image.open(o_image).convert('RGB')

        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
        # sample = {'image': image, 'question': qst2idc}
        if(self.phase=='valid'):
            sample = {'image': image, 'supporting_example_image': image, 'opposing_example_image': image, 'question': qst2idc}
        else:
            sample = {'image': image, 'supporting_example_image': s_image, 'opposing_example_image': o_image, 'question': qst2idc}

        if load_ans:
            ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']]
            ans2idx = np.random.choice(ans2idc)
            sample['answer_label'] = ans2idx         # for training

            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

        if transform:
            sample['image'] = transform(sample['image'])
            sample['supporting_example_image'] = transform(sample['supporting_example_image'])
            sample['opposing_example_image'] = transform(sample['opposing_example_image'])

        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):

    transform = {
        phase: transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))]) 
        for phase in ['train', 'valid']}

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            phase = 'train',
            transform=transform['train']),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            phase='valid',
            transform=transform['valid'])}

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader
