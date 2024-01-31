import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_mean_pool as gmp


class EncoderGIN(nn.Module):
    def __init__(self, input_dim=6, hid_dim=32, output_dim=8,re_train=False):
        super(EncoderGIN, self).__init__()
        nn1 = Sequential(Linear(input_dim, hid_dim, bias=True), ReLU(), Linear(hid_dim, hid_dim, bias=True))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        nn2_base = Sequential(Linear(hid_dim, hid_dim, bias=True), ReLU(), Linear(hid_dim, hid_dim, bias=True))
        self.conv2_base = GINConv(nn2_base)
        self.bn2_base = torch.nn.BatchNorm1d(hid_dim)
        nn3_base = Sequential(Linear(hid_dim, hid_dim,  bias=True), ReLU(), Linear(hid_dim, hid_dim,  bias=True))
        self.conv3_base = GINConv(nn3_base)
        self.bn3_base = torch.nn.BatchNorm1d(hid_dim)
        nn4_base = Sequential(Linear(hid_dim, hid_dim, bias=True), ReLU(), Linear(hid_dim, hid_dim, bias=True))
        self.conv4_base = GINConv(nn4_base)
        self.bn4_base = torch.nn.BatchNorm1d(hid_dim)
        nn5_base = Sequential(Linear(hid_dim, hid_dim, bias=True), ReLU(), Linear(hid_dim, hid_dim, bias=True))
        self.conv5_base = GINConv(nn5_base)
        self.bn5_base = torch.nn.BatchNorm1d(hid_dim)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, 2*hid_dim, bias=True),
            torch.nn.LeakyReLU(0.025),
            torch.nn.Linear(2 * hid_dim, hid_dim, bias=True),
            torch.nn.LeakyReLU(0.025),
            torch.nn.Linear(hid_dim, output_dim, bias=True)
        )
        self.re_train = re_train
        if self.re_train:
            self.output_layer = torch.nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu',a=1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch_data, t_sne=None):
        data_base,edge_index_base,batch_base=batch_data.x,batch_data.edge_index,batch_data.batch
        return self.forward_batch(data_base, edge_index_base, batch_base, t_sne=t_sne)

    def forward_batch(self, data_base, edge_index_base, batch_base, t_sne=False):
        x1_base = F.relu(self.conv1(data_base, edge_index_base))
        x1_base = self.bn1(x1_base)
        x2_base = F.relu(self.conv2_base(x1_base, edge_index_base))
        x2_base = self.bn2_base(x2_base)
        x3_base = F.relu(self.conv3_base(x2_base, edge_index_base))
        x3_base = self.bn3_base(x3_base)
        x_embedding_base = gmp(x3_base, batch_base)
        outputs = self.fc(x_embedding_base)
        if self.re_train:
            outputs = self.output_layer(outputs)
        if t_sne:
            return outputs, x_embedding_base
        else:
            return outputs

