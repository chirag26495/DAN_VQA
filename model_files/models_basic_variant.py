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

        # vggnet_feat = models.vgg19(pretrained=True).features
        vggnet_feat = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        modules = list(vggnet_feat.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Sequential(nn.Linear(self.cnn[-3].out_channels, embed_size),
                                nn.Tanh())     # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
    #     """
        with torch.no_grad():
            img_feature = self.cnn(image)                           # [batch_size, vgg16(19)_fc=4096]
        img_feature = img_feature.view(-1, 512, 196).transpose(1,2) # [batch_size, 196, 512]
        img_feature = self.fc(img_feature)                          # [batch_size, 196, embed_size]

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
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class NotAttention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        """Stacked attention Module
        """
        super(NotAttention, self).__init__()

        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_questions = nn.Linear(embed_size, num_channels)

        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.ff_attention = nn.Linear(num_channels, 1)

    def forward(self, vi, vq):
        """Extract feature vector from image vector.

        """
        hi = vi
        hq = vq.unsqueeze(dim=1)

        ha = torch.tanh(hi+hq)

        if self.dropout:
            ha = self.dropout(ha)

        u = ha.sum(dim=1)
        return u


class VqaModel(nn.Module):
    # num_attention_layer and num_mlp_layer not implemented yet
    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, stack_size=1):
        super(VqaModel, self).__init__()
        self.num_attention_layer = stack_size

        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)

        self.san = nn.ModuleList([NotAttention(512, embed_size)]*self.num_attention_layer)

        self.mlp = nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(embed_size, ans_vocab_size))

        self.tanh = nn.Tanh()

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]

        vi = img_feature
        u = qst_feature

        for not_attn_layer in self.san:
            u = not_attn_layer(vi, u)
            
        combined_feature = self.mlp(u)
        return combined_feature
