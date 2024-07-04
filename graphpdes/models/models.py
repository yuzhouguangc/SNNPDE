import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = Seq(
            Linear(in_size, hidden_size),
            Tanh(),
            Linear(hidden_size, 1, bias=True)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class Model(MessagePassing):
    def __init__(self, gamma, phi):
        super(Model, self).__init__(aggr='mean', flow='target_to_source')
        self.gamma = gamma
        self.phi = phi
        in_channels = 1 # fixed
        out_channels = 16 # tunable
        size_emb_dict = 3 # tunable
        self.mlp_adp = Seq(Linear(in_channels, out_channels),
                            ReLU(),
                            Linear(out_channels, 1))

        self.mlp_edge = Seq(Linear(in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, 1))

        self.mlp_tri = Seq(Linear(in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, 1))

        self.attention = Attention(1)
        self.embed_dict = nn.Parameter(torch.FloatTensor(725, size_emb_dict)) # size of nodes depends on the input graph
    def forward(self, u, edge_index, rel_pos, x_e, L1, x_tri, L2, B1, B2):
        return self.propagate(edge_index, u=u, rel_pos=rel_pos, x_e = x_e, L1=L1, x_tri = x_tri, L2 = L2, B1= B1, B2 = B2)

    def message(self, u_i, u_j, rel_pos):
        phi_input = torch.cat([u_i, u_j-u_i, rel_pos], dim=1)
        return self.phi(phi_input)

    def update(self, aggr, u, x_e, L1, x_tri, L2, B1, B2):
        gamma_input = torch.cat([u, aggr], dim=1)
        # the output of node-wise convolution
        node_dudt = self.gamma(gamma_input)
        # the output of adaptive node-wise convolution
        supports = F.softmax(F.relu(torch.mm(self.embed_dict, self.embed_dict.transpose(0, 1))), dim=1)
        # powers of supports
        # sec_hop_supports = torch.mm(supports, supports)
        # third_hop_supports = torch.mm(supports, sec_hop_supports)
        final_supports = supports # 0.5 * supports + 0.3 * sec_hop_supports + 0.2 * third_hop_supports
        adp_node_dudt = self.mlp_adp(torch.einsum('mm,md->md',final_supports, u))
        # the output of edge-wise convolution
        edge_dudt = self.mlp_edge(torch.einsum('mm,md->md',L1, x_e))
        edge_dudt_trans = torch.einsum('mn,md->nd', B1, edge_dudt)
        # the output of triangle-wise convolution
        triangle_dudt = self.mlp_tri(torch.einsum('qq,qd->qd',L2, x_tri))
        triangle_dudt_trans_ = torch.einsum('mq,qd->md', B2.T, triangle_dudt)
        triangle_dudt_trans = torch.einsum('nm,md->nd', B1.T, triangle_dudt_trans_)
        emb = torch.stack([edge_dudt_trans, triangle_dudt_trans], dim = 1)
        emb_dudt_trans, att = self.attention(emb)
        # alpha, beta, and gamma are hyperparameters
        # \sum{alpha, beta, gamma} = 1
        alpha = 0.5
        beta = 0.1
        gamma = 0.4
        dudt = alpha*emb_dudt_trans + beta*adp_node_dudt  + gamma*node_dudt
        return dudt


class ModelDirichlet(Model):
    def forward(self, u, edge_index, rel_pos, bcs_dict):
        return self.propagate(edge_index, u=u, rel_pos=rel_pos, bcs_dict=bcs_dict)
    
    def update(self, aggr, u, bcs_dict):
        dudt = super().update(aggr, u)
        for bc_inds, field_inds in bcs_dict.values():
            dudt[bc_inds, field_inds] *= 0  
        return dudt
