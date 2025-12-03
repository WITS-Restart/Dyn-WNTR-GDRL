import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchviz import make_dot
#from s2v_wdn_dqn.utils import visualize_pytorch_graph

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class QNetwork(nn.Module):
    def __init__(self, embedding_layers, n_node_features, n_edge_features, global_dim, embed_dim=256, bias=False, normalize=False):
        super().__init__()

        self.embedding_layers = embedding_layers
        self.embed_dim = embed_dim
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        self.node_features_embedding_layer = NodeFeaturesEmbeddingLayer(embed_dim, n_node_features, bias)
        self.edge_features_embedding_layer = EdgeFeaturesEmbeddingLayer(embed_dim, n_edge_features, bias)
        self.embedding_layer = EmbeddingLayer(embed_dim=embed_dim, bias=bias)
        self.global_fc = nn.Sequential(
            nn.Linear(global_dim, embed_dim, bias=True),
            nn.ReLU()
        )

        self.q_layer = DuelingScorerEdgeQLayer(n_edge_features=n_edge_features, embed_dim=embed_dim)

    def forward(self, state, edge_features, edges_ij, edge_status, global_feats):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        if self.n_edge_features > 0 and edge_features.dim() == 2:
            edge_features = edge_features.unsqueeze(0)
        
        node_features_embeddings = self.node_features_embedding_layer(state)
        embeddings = node_features_embeddings

        if edge_status is not None and not torch.is_tensor(edge_status):
            edge_status = torch.as_tensor(edge_status, device=state.device, dtype=state.dtype)

        for _ in range(self.embedding_layers):
            edge_features_embeddings = self.edge_features_embedding_layer(embeddings, edges_ij, edge_features, edge_status)
            embeddings = self.embedding_layer(embeddings, edges_ij, node_features_embeddings, edge_features_embeddings, edge_status)

        if not torch.is_tensor(global_feats):
            global_feats = torch.as_tensor(global_feats, device=state.device, dtype=state.dtype)
        if global_feats.dim() == 1:
            global_feats = global_feats.unsqueeze(0)   # (B,4)
        g_ctx = self.global_fc(global_feats)           # (B,D)
        g_ctx_exp = g_ctx.unsqueeze(1).expand(-1, embeddings.size(1), -1)  # (B,N,D)
        embeddings = torch.cat([embeddings, g_ctx_exp], dim=-1)  # (B,N,2D)

        #q_hat = self.q_layer(embeddings, edges_ij)
        return self.q_layer(embeddings, edge_features, edges_ij)



class DuelingScorerEdgeQLayer(nn.Module):
    """
    Dueling architecture for edge Q values.
    Output: Q values for all edges (B,E)
    
    Based on Wang et al. 2016: "Dueling Network Architectures for Deep Reinforcement Learning"
    https://arxiv.org/abs/1511.06581
    """


    def __init__(self, n_edge_features, embed_dim, use_norm=False, inner_dim=128):
        super().__init__()
        self.use_norm = use_norm
        self.state_proj = nn.Linear(2*embed_dim, embed_dim)
        self.action_proj = nn.Linear(n_edge_features, embed_dim)

        in_adv = 4*embed_dim
        self.adv_mlp = nn.Sequential(
            nn.ReLU(),
            NoisyLinear(in_adv, inner_dim),
            #nn.Linear(in_adv, inner_dim),
            nn.ReLU(),
            NoisyLinear(inner_dim, 1),
            #nn.Linear(inner_dim, 1)
        )
        
        self.attn = nn.Linear(2*embed_dim, 1)
        
        self.val_mlp = nn.Sequential(
            nn.ReLU(),
            NoisyLinear(embed_dim, inner_dim),
            #nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            NoisyLinear(inner_dim, 1),
            #nn.Linear(inner_dim, 1)
        )
        self.noop = NoisyLinear(embed_dim, 1)
        self.noop.bias_mu.data.fill_(-1.0)


    def forward(self, graph_embeddings, edge_features, edges_ij):
        B, N, D2 = graph_embeddings.shape
        B2, E, F = edge_features.shape

        scores = self.attn(graph_embeddings).squeeze(-1)      # (B, N)
        alpha  = torch.softmax(scores, dim=1)                # (B, N)
        g = (alpha.unsqueeze(-1) * graph_embeddings).sum(1)   # (B, 2D)
        g = self.state_proj(g)                                  # (B, D)

        i_idx = edges_ij[:, 0]                                  # (E,)
        j_idx = edges_ij[:, 1]                                  # (E,)
        mu_i  = graph_embeddings[:, i_idx, :]                    # (B, E, 2D)
        mu_j  = graph_embeddings[:, j_idx, :]                    # (B, E, 2D)
        mu_i = self.state_proj(mu_i)                            # (B, E, D)
        mu_j = self.state_proj(mu_j)                            # (B, E, D)

        a = self.action_proj(edge_features.reshape(B*E, F)).reshape(B, E, -1)  # (B, E, D)

        g_tiled = g.unsqueeze(1).expand(-1, E, -1)              # (B, E, D)
        sa = torch.cat([g_tiled, mu_i, mu_j, a], dim=-1)        # (B, E, 4D)
        adv = self.adv_mlp(sa).squeeze(-1)                      # (B, E)

        val = self.val_mlp(g).squeeze(-1)                       # (B,)

        q = val.unsqueeze(1) + adv - adv.mean(dim=1, keepdim=True)  # (B, E)
        noop_q = self.noop(g)

        q_all = torch.cat([q, noop_q], dim=1)

        return q
    
class NodeFeaturesEmbeddingLayer(nn.Module):
    """
    Calculate the theta1 component
    """
    def __init__(self, embed_dim, n_node_features, bias=False):
        super().__init__()
        self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias)

    def forward(self, node_features):
        return self.theta1(node_features)

class EdgeFeaturesEmbeddingLayer(nn.Module):
    """
    Calculate the theta3/theta4 component
    """
    def __init__(self, embed_dim, n_edge_features, bias=False):
        super().__init__()
        self.theta3 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta4 = nn.Linear(n_edge_features, embed_dim, bias=bias)

    def forward(self, embeddings, edges_ij, edge_features, edge_status):
        """
        embeddings:    (B, N, D)
        edges_ij:      (E, 2) long tensor (canonical undirected edges)
        edge_features: (B, E, F_edge)
        returns:       (B, N, D) edge-driven contribution
        """
        B, N, D = embeddings.shape
        u = edges_ij[:, 0]
        v = edges_ij[:, 1]

        x4 = F.leaky_relu(self.theta4(edge_features))      # (B, E, D)

        g = edge_status
        if g.dim() == 1:            # (E,)
            g = g.unsqueeze(0)      # (1,E)
        if g.dim() == 2:            # (B,E)
            g = g.unsqueeze(-1)     # (B,E,1)
        if g.size(0) == 1 and B > 1:
            g = g.expand(B, -1, -1) # broadcast across batch
        g = g.to(dtype=x4.dtype, device=x4.device)

        x4 = x4 * g  # gate edge messages

        msg = torch.zeros(B, N, D, device=x4.device, dtype=x4.dtype)
        msg.scatter_add_(1, u.view(1, -1, 1).expand(B, -1, D), x4)
        msg.scatter_add_(1, v.view(1, -1, 1).expand(B, -1, D), x4)

        return self.theta3(msg)                            # (B, N, D)

class EmbeddingLayer(nn.Module):
    """
    Calculate embeddings for all nodes
    """
    def __init__(self, embed_dim, bias=False):
        super().__init__()
        #self.theta1 = nn.Linear(n_node_features, embed_dim, bias=bias) # now in NodeFeaturesEmbeddingLayer
        self.theta2 = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, prev_embeddings, edges_ij, node_features_embeddings, edge_features_embeddings, edge_status):
        """
        prev_embeddings:         (B, N, D)
        edges_ij:                (E, 2)
        node_features_embeddings:(B, N, D)  [your theta1(node_feats)]
        edge_features_embeddings:(B, N, D)  [from EdgeFeaturesEmbeddingLayer]
        edge_status:             None or (E,), (B,E) or (B,E,1) in [0,1]

        """
        B, N, D = prev_embeddings.shape
        u = edges_ij[:, 0]
        v = edges_ij[:, 1]

        g = edge_status
        if g.dim() == 1:            # (E,)
            g = g.unsqueeze(0)      # (1,E)
        if g.dim() == 2:            # (B,E)
            g = g.unsqueeze(-1)     # (B,E,1)
        if g.size(0) == 1 and B > 1:
            g = g.expand(B, -1, -1)
        g = g.to(dtype=prev_embeddings.dtype, device=prev_embeddings.device)

        # neighbor sum of node embeddings (two-way since undirected)
        #nbr = prev_embeddings.new_zeros(B, N, D)
        #nbr.index_add_(1, u, prev_embeddings[:, v, :])     # j -> i
        #nbr.index_add_(1, v, prev_embeddings[:, u, :])     # i -> j

        nbr = torch.zeros(B, N, D, device=prev_embeddings.device, dtype=prev_embeddings.dtype)
        h_u = prev_embeddings[:, u, :] * g
        h_v = prev_embeddings[:, v, :] * g
        nbr.scatter_add_(1, v.view(1, -1, 1).expand(B, -1, D), h_u)
        nbr.scatter_add_(1, u.view(1, -1, 1).expand(B, -1, D), h_v)

        x2 = self.theta2(nbr)
        ret = F.leaky_relu(node_features_embeddings + x2 + edge_features_embeddings)
        return ret

class NoisyLinear():
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__(in_features, out_features)
        # Standard parameters (µ)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_noisy_parameters()

    def reset_noisy_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_sigma.data.fill_(self.sigma_init)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return torch.nn.functional.linear(x, w, b)

    def sample_noise(self):
        with torch.no_grad():
            self.weight_epsilon.normal_()
            self.bias_epsilon.normal_()

    def remove_noise(self):
        with torch.no_grad():
            self.weight_epsilon.zero_()
            self.bias_epsilon.zero_()




''' --- IGNORE ---
class OldEdgeQLayer(nn.Module):
    def __init__(self, embed_dim, bias=False, normalize=False):
        super().__init__()
        self.theta5 = nn.Linear(3*embed_dim, 1, bias=bias)  # [global, i, j]
        self.theta6 = nn.Linear(embed_dim, embed_dim, bias=bias)  # global proj
        self.theta7 = nn.Linear(embed_dim, embed_dim, bias=bias)  # node proj
        #self.noop   = nn.Linear(embed_dim, 1, bias=bias)
        self.normalize = normalize

    def forward(self, embeddings, edges_ij):
        # embeddings: (B,N,D); edges_ij: (E,2) long
        B, N, D = embeddings.shape
        u = edges_ij[:,0]
        v = edges_ij[:,1]

        g = embeddings.sum(1)                        # (B,D)
        if self.normalize: g = g / N
        g_proj = self.theta6(g)                      # (B,D)

        node_proj = self.theta7(embeddings)          # (B,N,D)
        i_proj = node_proj[:, u, :]                  # (B,E,D)
        j_proj = node_proj[:, v, :]                  # (B,E,D)
        g_tile = g_proj.unsqueeze(1).expand(-1, i_proj.size(1), -1)  # (B,E,D)

        feats = torch.cat([g_tile, i_proj, j_proj], dim=-1)          # (B,E,3D)
        feats = torch.nn.functional.leaky_relu(feats)
        edge_q = self.theta5(feats).squeeze(-1)                       # (B,E)
        #noop_q = self.noop(g)                                         # (B,1)
        #return torch.cat([edge_q, noop_q], dim=1)                     # (B,E+1)
        return edge_q                     # (B,E+1)

class OldQLayer(nn.Module):
    """
    Given node embeddings, calculate Q_hat for all vertices
    """
    def __init__(self, embed_dim, bias=False, normalize=False):
        super().__init__()
        self.theta5 = nn.Linear(3*embed_dim, 1, bias=bias)
        self.theta6 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.theta7 = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.noop = nn.Linear(embed_dim, 1, bias=bias)  # No-op layer for compatibility
        self.normalize = normalize

    def forward(self, embeddings):
        # embeddings: (batch_size, N, D)
        B, N, D = embeddings.shape

        # 1) Global summary
        g = embeddings.sum(dim=1)                  # (B, D)
        if self.normalize:
            g = g / N
        g_proj = self.theta6(g)                    # (B, D)

        # 2) Pairwise i→j embeddings
        #    i_embed[b,i,j] = embeddings[b,i,:]
        i_embed = embeddings.unsqueeze(2).expand(-1, N, N, -1)  # (B,N,N,D)
        #    j_embed[b,i,j] = embeddings[b,j,:]
        j_embed = embeddings.unsqueeze(1).expand(-1, N, N, -1)  # (B,N,N,D)

        # 3) Project them
        i_proj = self.theta7(i_embed)              # (B,N,N,D)
        j_proj = self.theta7(j_embed)              # (B,N,N,D)

        # 4) Tile global proj over all (i,j)
        g_tile = g_proj.view(B,1,1,D).expand(-1, N, N, -1)  # (B,N,N,D)

        # 5) Concatenate [g_tile, i_proj, j_proj] → (B,N,N,3D)
        features = torch.cat([g_tile, i_proj, j_proj], dim=-1)
        features = nn.LeakyReLU()(features)

        # 6) Compute edge-Q and flatten to (B, N*N)
        edge_q = self.theta5(features).squeeze(-1).view(B, -1)  # (B, N, N) -> (B, N*N) 
        # 7) Compute no-op Q → (B,1)
        noop_q = self.noop(g)                      # (B, 1)

        # 8) Final vector → (B, N*N + 1)
        return torch.cat([edge_q, noop_q], dim=1)


    def old_forward(self, embeddings):
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
        
        return out        

class OldEmbeddingLayer(nn.Module):
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

class OldEdgeFeaturesEmbeddingLayer(nn.Module):
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
'''