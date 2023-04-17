import torch
import torch.nn as nn
import torchvision.models as models


class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()

        # model = models.vgg19(pretrained=True)
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        modules = list(model.features.children())[:-1]
        # modules = list(model.features.children())[:-2]
        self.features = nn.Sequential(*modules)    # remove last maxpool layer
        
        # self.classifier = model.classifier

        in_features = self.features[-3].out_channels # input size of feature vector
        # in_features = model.features[-5].out_channels # input size of feature vector

        self.fc = nn.Linear(in_features, embed_size)
        # self.fc = nn.Sequential(
        #         nn.Linear(in_features, embed_size),
        #         nn.Tanh())

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        
        with torch.no_grad():
            img_feature = self.features(image)                  # [batch_size, vgg16(19)_fc=4096]

        ## Flattening out the image part (not with channels)
        img_feature = img_feature.view(-1, 512, 196).transpose(1, 2) # [batch_size, 196, 512]

        # with torch.no_grad():
        #     img_feature = self.classifier(img_feature)

        img_feature = self.fc(img_feature)                   # [batch_size, 196, embed_size]

        ## Normalizing Ouput
        # l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        # l2_norm = img_feature.norm(p=2, dim=2, keepdim=True).detach()

        # img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512] qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]

        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     # [batch_size, 196, embed_size

        # qst_feature = self.qst_encoder(qst)                     # [batch_size, 196, embed_size]
        qst_feature = self.qst_encoder(qst).unsqueeze(dim=1)                     # [batch_size, 196, embed_size]

        # combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = (img_feature + qst_feature).sum(dim=1)  # [batch_size, embed_size]

        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)

        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)

        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature
