import itertools
import logging
import sys

import torch
import torch.nn as nn

from s2v_dqn.model import visualize_pytorch_graph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, embedding_layers, n_node_features, n_edge_features, embed_dim=64, bias=False, normalize=False):
        super().__init__()

        self.embedding_layers = embedding_layers
        self.embed_dim = embed_dim
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        self.node_features_embedding_layer = NodeFeaturesEmbeddingLayer(embed_dim, n_node_features, bias)
        self.edge_features_embedding_layer = EdgeFeaturesEmbeddingLayer(embed_dim, n_edge_features, bias, normalize)
        self.embedding_layer = EmbeddingLayer(embed_dim=embed_dim, bias=bias, normalize=normalize)

        self.q_layer = QLayer(embed_dim=embed_dim, bias=bias, normalize=normalize)

    def forward(self, state, edge_features):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if self.n_edge_features > 0 and edge_features.dim() == 3:
            edge_features = edge_features.unsqueeze(0)

        node_features = state[:, :, :self.n_node_features]
        adj = state[:, :, self.n_node_features:]
        # adj = state[:, :, self.n_node_features:(self.n_node_features + n_vertices)]
        # edge_features = state[:, :, (self.n_node_features + n_vertices):]

        # calculate node embeddings
        embeddings = torch.zeros(state.shape[0], state.shape[1], self.embed_dim, requires_grad=True).to(device, dtype=torch.float32)
        node_features_embeddings = self.node_features_embedding_layer(node_features)
        edge_features_embeddings = self.edge_features_embedding_layer(edge_features, adj) if self.n_edge_features > 0 else None
        # print(f"{edge_features_embeddings=}")

        for _ in range(self.embedding_layers):
            embeddings = self.embedding_layer(embeddings, adj, node_features_embeddings, edge_features_embeddings)

        # visualize_pytorch_graph(self.q_layer, embeddings)

        # make_dot(
        #     self.q_layer(embeddings),
        #     params=dict(itertools.chain(
        #         self.q_layer.named_parameters(),
        #         self.node_features_embedding_layer.named_parameters(),
        #         self.edge_features_embedding_layer.named_parameters(),
        #         self.embedding_layer.named_parameters(),
        #     ))
        # ).render(f"torchviz_QLayer2", format="png")
        # sys.exit(0)

        # calculate \hat{Q} based on embeddings and given vertices
        q_hat = self.q_layer(embeddings)
        return q_hat


class NodeFeaturesEmbeddingLayer(nn.Module):
    """
    Calculate the theta1 component
    """
    def __init__(self, embed_dim, n_node_features, bias=False):
        super().__init__()
        self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)

    def forward(self, node_features):
        # node_features.shape = (batch_size, n_vertices, n_node_features)
        # ret.shape = (batch_size, n_vertices, embed_dim)
        ret = self.theta1(node_features)
        return ret


class EdgeFeaturesEmbeddingLayer(nn.Module):
    """
    Calculate the theta3/theta4 component
    """
    def __init__(self, embed_dim, n_edge_features, bias=False, normalize=False):
        super().__init__()
        self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias)
        self.normalize = normalize

    def forward(self, edge_features, adj):
        # edge_features.shape = (batch_size, n_vertices, n_vertices, n_edge_features)
        # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        if edge_features.dim() == 3:
            logging.warning("Wrong number of dimensions")

        # x4 = F.relu(self.theta4(edge_features))
        x4 = nn.LeakyReLU()(self.theta4(edge_features))

        # adj.shape = (batch_size, n_vertices, n_vertices)
        # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        # sum_neighbor_edge_embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x3.shape = (batch_size, n_vertices, embed_dim)
        sum_neighbor_edge_embeddings = (adj.unsqueeze(-1) * x4).sum(dim=2)

        if self.normalize:
            norm = adj.sum(dim=2).unsqueeze(-1)
            norm[norm == 0] = 1
            sum_neighbor_edge_embeddings = sum_neighbor_edge_embeddings / norm

        ret = self.theta3(sum_neighbor_edge_embeddings)

        return ret


class EmbeddingLayer(nn.Module):
    """
    Calculate embeddings for all vertices
    """
    def __init__(self, embed_dim, bias=False, normalize=False):
        super().__init__()
        # self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)
        self.theta2 = nn.Linear(embed_dim, embed_dim, bias=bias, )
        # self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias) if n_edge_features > 0 else None
        self.normalize = normalize
        
    def forward(self, prev_embeddings, adj, node_features_embeddings, edge_features_embeddings):
        # adj.shape = (batch_size, n_vertices, n_vertices)
        # prev_embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x2.shape = (batch_size, n_vertices, embed_dim)
        x2 = self.theta2(torch.matmul(adj, prev_embeddings))

        x1 = node_features_embeddings
        x3 = edge_features_embeddings

        if x3 is not None:
            ret = nn.LeakyReLU()(x1 + x2 + x3)
        else:
            ret = nn.LeakyReLU()(x1 + x2)

        return ret

        # node_features.shape = (batch_size, n_vertices, n_node_features)
        # x1.shape = (batch_size, n_vertices, embed_dim)
        # x1 = self.theta1(node_features)

        # n_edge_features = edge_features.shape[2]
        # if n_edge_features > 0:
        #     # edge_features.shape = (batch_size, n_vertices, n_vertices, n_edge_features)
        #     # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        #     if edge_features.dim() == 3:
        #         edge_features = edge_features.unsqueeze(-1)
        #     # x4 = F.relu(self.theta4(edge_features))
        #     x4 = nn.LeakyReLU()(self.theta4(edge_features))
        #
        #     # adj.shape = (batch_size, n_vertices, n_vertices)
        #     # x4.shape = (batch_size, n_vertices, n_vertices, embed_dim)
        #     # sum_neighbor_edge_embeddings.shape = (batch_size, n_vertices, embed_dim)
        #     # x3.shape = (batch_size, n_vertices, embed_dim)
        #     sum_neighbor_edge_embeddings = (adj.unsqueeze(-1) * x4).sum(dim=2)
        #     if self.normalize:
        #         norm = adj.sum(dim=2).unsqueeze(-1)
        #         norm[norm == 0] = 1
        #         sum_neighbor_edge_embeddings = sum_neighbor_edge_embeddings / norm
        #
        #     x3 = self.theta3(sum_neighbor_edge_embeddings)
        #
        #     ret = nn.LeakyReLU()(x1 + x2 + x3)
        # else:
        #     ret = nn.LeakyReLU()(x1 + x2)

        # ret.shape = (batch_size, n_vertices, embed_dim)
        # ret = F.relu(x1 + x2 [+ x3])
        # return ret


class QLayer(nn.Module):
    """
    Given node embeddings, calculate Q_hat for all vertices
    """
    def __init__(self, embed_dim, bias=False, normalize=False):
        super().__init__()
        self.theta5 = nn.Linear(2*embed_dim, 1, bias=bias)
        self.theta6 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta7 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.normalize = normalize

    def forward(self, embeddings):
        # embeddings.shape = (batch_size, n_vertices, embed_dim)
        # sum_embeddings.shape = (batch_size, embed_dim)
        # x6.shape = (batch_size, embed_dim)
        sum_embeddings = embeddings.sum(dim=1)
        if self.normalize:
            sum_embeddings = sum_embeddings / embeddings.shape[1]
        x6 = self.theta6(sum_embeddings)
        
        # repeat graph embedding for all vertices
        # x6.shape = (batch_size, embed_dim)
        # embeddings.shape[1] = n_vertices
        # x6_repeated.shape = (batch_size, n_vertices, embed_dim)
        x6_repeated = x6.unsqueeze(1).repeat(1, embeddings.shape[1], 1)
        
        # embeddings.shape = (batch_size, n_vertices, embed_dim)
        # x7.shape = (batch_size, n_vertices, embed_dim)
        x7 = self.theta7(embeddings)
        
        # x6.shape = x7.shape = (batch_size, n_vertices, embed_dim)
        # features.shape = (batch_size, n_vertices, 2*embed_dim)
        # x5.shape = (batch_size, n_vertices, 1)
        # features = F.relu(torch.cat([x6_repeated, x7], dim=-1))
        features = nn.LeakyReLU()(torch.cat([x6_repeated, x7], dim=-1))
        x5 = self.theta5(features)
        
        # out.shape = (batch_size, n_vertices)
        out = x5.squeeze(-1)

        # print(f"{out.shape=}")
        
        return out        
