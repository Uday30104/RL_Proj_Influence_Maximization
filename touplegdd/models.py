import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_softmax, scatter_max
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
import utils.graph_utils as graph_utils
# Import the fixed community feature dimension.
# COMM_FEAT_DIM = 5 always, regardless of how many communities a graph has.
# The projection layers are always Linear(5, embed_dim), so weights transfer
# cleanly from training graphs (12 communities) to test graphs (3191 communities).
from utils.graph_utils import COMM_FEAT_DIM
from collections import deque
from tqdm import tqdm

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

EPS = 1e-15


class S2V_DQN(nn.Module):
    ''' Structure2Vec DQN baseline '''
    def __init__(self, reg_hidden, embed_dim, node_dim, edge_dim, T, w_scale, avg=False, num_communities=0):
        '''w_scale=0.01, node_dim=2, edge_dim=4'''
        super(S2V_DQN, self).__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.avg = avg

        self.num_communities = num_communities
        if self.num_communities > 0:
            # Input is always COMM_FEAT_DIM (5), not the number of communities
            # in the graph.  This is the key fix: the weight matrix is 5×embed_dim
            # on both training and test graphs.
            self.comm_proj = nn.Linear(COMM_FEAT_DIM, embed_dim, bias=True)
            torch.nn.init.normal_(self.comm_proj.weight, mean=0, std=w_scale)
            torch.nn.init.zeros_(self.comm_proj.bias)

        self.w_n2l = torch.nn.Parameter(torch.Tensor(node_dim, embed_dim))
        torch.nn.init.normal_(self.w_n2l, mean=0, std=w_scale)

        self.w_e2l = torch.nn.Parameter(torch.Tensor(edge_dim, embed_dim))
        torch.nn.init.normal_(self.w_e2l, mean=0, std=w_scale)

        self.p_node_conv = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.p_node_conv, mean=0, std=w_scale)

        self.trans_node_1 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_1, mean=0, std=w_scale)

        self.trans_node_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_2, mean=0, std=w_scale)

        if self.reg_hidden > 0:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, reg_hidden))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.h2_weight = torch.nn.Parameter(torch.Tensor(reg_hidden, 1))
            torch.nn.init.normal_(self.h2_weight, mean=0, std=w_scale)
            self.last_w = self.h2_weight
        else:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.last_w = self.h1_weight

        self.scatter_aggr = (scatter_mean if self.avg else scatter_add)

    def forward(self, data):
        data.x = torch.matmul(data.x, self.w_n2l)

        # Inject community features into the initial node embedding.
        # data.comm has shape (total_nodes_in_batch, COMM_FEAT_DIM=5).
        # Column 4 (novelty flag) has already been filled by rl_agents.py
        # based on the current seed set state.
        if hasattr(data, 'comm') and self.num_communities > 0:
            data.x = data.x + self.comm_proj(data.comm)

        data.x = F.relu(data.x)

        data.edge_attr = torch.matmul(data.edge_attr, self.w_e2l)

        for _ in range(self.T):
            msg_linear = torch.matmul(data.x, self.p_node_conv)
            n2e_linear = msg_linear[data.edge_index[0]]

            edge_rep = torch.add(n2e_linear, data.edge_attr)
            edge_rep = F.relu(edge_rep)

            e2n = self.scatter_aggr(edge_rep, data.edge_index[1], dim=0,
                                    dim_size=data.x.size(0))

            data.x = torch.add(torch.matmul(e2n, self.trans_node_1),
                                torch.matmul(data.x, self.trans_node_2))
            data.x = F.relu(data.x)

        y_potential = self.scatter_aggr(data.x, data.batch, dim=0)

        if data.y is not None:  # Q func given a specific action
            action_embed = data.x[data.y]
            embed_s_a = torch.cat((action_embed, y_potential), dim=-1)

            last_output = embed_s_a
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = F.relu(hidden)
            q_pred = torch.matmul(last_output, self.last_w)
            return q_pred

        else:  # Q func on all nodes simultaneously
            rep_y = y_potential[data.batch]
            embed_s_a_all = torch.cat((data.x, rep_y), dim=-1)

            last_output = embed_s_a_all
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                last_output = torch.relu(hidden)

            q_on_all = torch.matmul(last_output, self.last_w)
            return q_on_all


class Tripling(nn.Module):
    """
    ToupleGDD: three coupled GNNs (state, source, target) + DDQN.

    Community feature injection:
      At the very start of forward(), the community feature vector (5-dim) is
      projected onto embed_dim and added to both the source and target initial
      embeddings.  Column 4 of the feature vector (the dynamic novelty flag)
      varies per step based on which communities the current seed set has already
      covered — this is what lets the GNN learn to value community diversity.

      The projection weights are always 5×embed_dim (COMM_FEAT_DIM × embed_dim),
      fixed regardless of how many communities exist in the graph.
    """
    def __init__(self, embed_dim, sgate_l1_dim, tgate_l1_dim, T, hidden_dims,
                 w_scale, num_communities=0):
        super(Tripling, self).__init__()
        self.embed_dim = embed_dim
        self.sgate_l1_dim = sgate_l1_dim
        self.tgate_l1_dim = tgate_l1_dim
        self.T = T
        self.hidden_dims = hidden_dims.copy()
        self.hidden_dims.insert(0, embed_dim)  # prepend initial dim

        self.num_communities = num_communities
        if self.num_communities > 0:
            # Both projections are COMM_FEAT_DIM → embed_dim.
            # Source projection: encodes how much community-spread capacity
            #                    this node has (learned from columns 0-3 +
            #                    novelty signal from column 4).
            # Target projection: encodes how receptive this node's neighbourhood
            #                    is to novel-community spread.
            self.comm_proj_source = nn.Linear(COMM_FEAT_DIM, embed_dim, bias=True)
            self.comm_proj_target = nn.Linear(COMM_FEAT_DIM, embed_dim, bias=True)
            torch.nn.init.normal_(self.comm_proj_source.weight, mean=0, std=w_scale)
            torch.nn.init.normal_(self.comm_proj_target.weight, mean=0, std=w_scale)
            torch.nn.init.zeros_(self.comm_proj_source.bias)
            torch.nn.init.zeros_(self.comm_proj_target.bias)

        # ---- per-layer parameter lists ----
        self.trans_weights = nn.ParameterList()

        # state GNN
        self.influgate_etas = nn.ParameterList()
        self.state_weights_self = nn.ParameterList()
        self.state_weights_neibor = nn.ParameterList()
        self.state_weights_attention = nn.ParameterList()
        self.state_weights_edge = nn.ParameterList()

        # source GNN
        self.source_betas = nn.ParameterList()
        self.sourcegate_layer1s = nn.ModuleList()
        self.sourcegate_layer2s = nn.ModuleList()
        self.source_weights_self = nn.ParameterList()
        self.source_weights_neibor = nn.ParameterList()
        self.source_weights_state = nn.ParameterList()
        self.source_weights_attention = nn.ParameterList()
        self.source_weights_edge = nn.ParameterList()

        # target GNN
        self.target_taus = nn.ParameterList()
        self.targetgate_layer1s = nn.ModuleList()
        self.targetgate_layer2s = nn.ModuleList()
        self.target_weights_self = nn.ParameterList()
        self.target_weights_neibor = nn.ParameterList()
        self.target_weights_state = nn.ParameterList()
        self.target_weights_attention = nn.ParameterList()
        self.target_weights_edge = nn.ParameterList()

        for i in range(1, T + 1):
            h_prev = self.hidden_dims[i - 1]
            h_curr = self.hidden_dims[i]

            self.trans_weights.append(
                torch.nn.Parameter(torch.Tensor(h_prev, h_curr)))
            torch.nn.init.normal_(self.trans_weights[-1], mean=0, std=w_scale)

            # state GNN
            self.influgate_etas.append(
                torch.nn.Parameter(torch.Tensor(2 * h_curr, 1)))
            torch.nn.init.normal_(self.influgate_etas[-1], mean=0, std=w_scale)
            self.state_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_self[-1], mean=0, std=w_scale)
            self.state_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_neibor[-1], mean=0, std=w_scale)
            self.state_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_attention[-1], mean=0, std=w_scale)
            self.state_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_edge[-1], mean=0, std=w_scale)

            # source GNN
            self.source_betas.append(
                torch.nn.Parameter(torch.Tensor(2 * h_curr, 1)))
            torch.nn.init.normal_(self.source_betas[-1], mean=0, std=w_scale)
            self.sourcegate_layer1s.append(
                torch.nn.Linear(h_prev, sgate_l1_dim, True))
            self.sourcegate_layer2s.append(
                torch.nn.Linear(sgate_l1_dim, h_curr, True))
            self.source_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_self[-1], mean=0, std=w_scale)
            self.source_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_neibor[-1], mean=0, std=w_scale)
            self.source_weights_state.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_state[-1], mean=0, std=w_scale)
            self.source_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_attention[-1], mean=0, std=w_scale)
            self.source_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_edge[-1], mean=0, std=w_scale)

            # target GNN
            self.target_taus.append(
                torch.nn.Parameter(torch.Tensor(2 * h_curr, 1)))
            torch.nn.init.normal_(self.target_taus[-1], mean=0, std=w_scale)
            self.targetgate_layer1s.append(
                torch.nn.Linear(h_prev, tgate_l1_dim, True))
            self.targetgate_layer2s.append(
                torch.nn.Linear(tgate_l1_dim, h_curr, True))
            self.target_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_self[-1], mean=0, std=w_scale)
            self.target_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_neibor[-1], mean=0, std=w_scale)
            self.target_weights_state.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_state[-1], mean=0, std=w_scale)
            self.target_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_attention[-1], mean=0, std=w_scale)
            self.target_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_edge[-1], mean=0, std=w_scale)

        # DQN scoring head — equation (14) of the paper
        self.theta1 = torch.nn.Parameter(torch.Tensor(3 * self.hidden_dims[-1], 1))
        torch.nn.init.normal_(self.theta1, mean=0, std=w_scale)
        self.theta2 = torch.nn.Parameter(
            torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta2, mean=0, std=w_scale)
        self.theta3 = torch.nn.Parameter(
            torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta3, mean=0, std=w_scale)
        self.theta4 = torch.nn.Parameter(
            torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta4, mean=0, std=w_scale)

    def forward(self, data):
        """
        Forward pass: inject community features, run ToupleGNN for T layers,
        then compute Q-values.

        data.x layout: [:embed_dim] = PDW source embedding S
                        [embed_dim:2*embed_dim] = PDW target embedding T
                        [-1] = activation state (1 if selected seed, else 0)
        data.comm:      shape (total_nodes, COMM_FEAT_DIM=5)
                        column 4 is the dynamic novelty flag filled per-step
        """
        source_influ = data.x[:, :self.hidden_dims[0]]
        target_influ = data.x[:, self.hidden_dims[0]:2 * self.hidden_dims[0]]
        state = data.x[:, -1]

        # Inject community features into initial source and target embeddings.
        # The novelty flag (column 4) carries the episode-level signal about
        # how many of this node's communities are still uncovered by seeds.
        # The GNN then propagates this flag through T hops, building up an
        # estimate of community-spread potential for each candidate node.
        if hasattr(data, 'comm') and self.num_communities > 0:
            source_influ = source_influ + F.leaky_relu(
                self.comm_proj_source(data.comm), 0.2)
            target_influ = target_influ + F.leaky_relu(
                self.comm_proj_target(data.comm), 0.2)

        for i in range(self.T):
            trans_source = torch.matmul(source_influ, self.trans_weights[i])
            trans_target = torch.matmul(target_influ, self.trans_weights[i])
            # For each edge (u→v): concat [trans_source_u, trans_target_v]
            trans_influ = torch.cat(
                (trans_source[data.edge_index[0]],
                 trans_target[data.edge_index[1]]), dim=-1)

            # ---- State GNN (equations 5-7 of paper) ----
            e_uv = torch.matmul(trans_influ, self.influgate_etas[i]).squeeze(1)
            e_uv = F.leaky_relu(e_uv, 0.2)
            influgate = scatter_softmax(e_uv, data.edge_index[1])
            influgate = (influgate * self.state_weights_attention[i]
                         + data.edge_weight * self.state_weights_edge[i])
            a_v = scatter_add(influgate * state[data.edge_index[0]],
                              data.edge_index[1], dim_size=data.x.size(0))
            new_state = torch.sigmoid(
                state * self.state_weights_self[i]
                + a_v * self.state_weights_neibor[i])
            # Seeds stay locked at state=1
            new_state = new_state * (1 - data.x[:, -1]) + data.x[:, -1]

            # ---- Source GNN (equations 8-10 of paper) ----
            f_vw = torch.matmul(trans_influ, self.source_betas[i]).squeeze(1)
            f_vw = F.leaky_relu(f_vw, 0.2)
            alpha_vw = scatter_softmax(f_vw, data.edge_index[0])
            alpha_vw = (alpha_vw * self.source_weights_attention[i]
                        + data.edge_weight * self.source_weights_edge[i])
            sourcegate = F.leaky_relu(
                self.sourcegate_layer2s[i](
                    F.leaky_relu(
                        self.sourcegate_layer1s[i](
                            target_influ[data.edge_index[1]]), 0.2)), 0.2)
            b_v = scatter_add(alpha_vw.unsqueeze(1) * sourcegate,
                              data.edge_index[0], dim=0,
                              dim_size=data.x.size(0))
            new_source = F.leaky_relu(
                trans_source * self.source_weights_self[i]
                + b_v * self.source_weights_neibor[i]
                + (state * self.source_weights_state[i]).unsqueeze(1))

            # ---- Target GNN (equations 11-13 of paper) ----
            d_uv = torch.matmul(trans_influ, self.target_taus[i]).squeeze(1)
            d_uv = F.leaky_relu(d_uv, 0.2)
            phi_uv = scatter_softmax(d_uv, data.edge_index[1])
            phi_uv = (phi_uv * self.target_weights_attention[i]
                      + data.edge_weight * self.target_weights_edge[i])
            targetgate = F.leaky_relu(
                self.targetgate_layer2s[i](
                    F.leaky_relu(
                        self.targetgate_layer1s[i](
                            source_influ[data.edge_index[0]]), 0.2)), 0.2)
            c_v = scatter_add(phi_uv.unsqueeze(1) * targetgate,
                              data.edge_index[1], dim=0,
                              dim_size=data.x.size(0))
            new_target = F.leaky_relu(
                trans_target * self.target_weights_self[i]
                + c_v * self.target_weights_neibor[i]
                + (state * self.target_weights_state[i]).unsqueeze(1))

            state = new_state
            source_influ = new_source
            target_influ = new_target

        # ---- Q-value computation (equation 14 of paper) ----
        if data.y is not None:  # training mode: Q for a specific action
            S_v = source_influ[data.y]

            not_y = torch.ones(target_influ.size(0), dtype=torch.bool,
                               device=data.y.device)
            not_y[data.y] = False
            not_selected = data.x[:, -1] == 0
            not_idx = torch.logical_and(not_y, not_selected)

            batch_idx = data.batch[not_idx]
            T_u = target_influ[not_idx]

            is_idx = data.x[:, -1] == 1
            batch_is_idx = data.batch[is_idx]
            S_w = source_influ[is_idx]

            batch_size = data.batch[-1].item() + 1
            q_pred = torch.matmul(
                F.leaky_relu(torch.cat([
                    torch.matmul(S_v, self.theta2),
                    torch.matmul(
                        scatter_add(S_w, batch_is_idx, dim=0,
                                    dim_size=batch_size),
                        self.theta4),
                    torch.matmul(
                        scatter_add(T_u, batch_idx, dim=0,
                                    dim_size=batch_size),
                        self.theta3),
                ], dim=-1)),
                self.theta1)
            return q_pred

        else:  # inference mode: Q for all candidate nodes simultaneously
            target_influ[data.x[:, -1] == 1] = 0.0
            state[data.x[:, -1] == 1] = 1.0

            source_influ_copy = source_influ.clone()
            source_influ[data.x[:, -1] == 0] = 0.0
            counts = scatter_add(
                torch.ones(data.batch.size(0), dtype=torch.long,
                           device=data.batch.device),
                data.batch)
            source_influ_w = (scatter_add(source_influ, data.batch, dim=0)
                              .repeat_interleave(counts, dim=0))
            source_influ_w[data.x[:, -1] == 1] = 0.0
            source_influ_copy[data.x[:, -1] == 1] = 0.0

            target_sum = (scatter_add(target_influ, data.batch, dim=0)
                          .repeat_interleave(counts, dim=0))

            q_on_all = torch.matmul(
                F.leaky_relu(torch.cat([
                    torch.matmul(source_influ_copy, self.theta2),
                    torch.matmul(source_influ_w, self.theta4),
                    torch.matmul(target_sum - target_influ, self.theta3),
                ], dim=-1)),
                self.theta1)
            return q_on_all


# ----------------------------------------------------------------------
# Personalized DeepWalk (PDW) — initial embedding pre-training
# ----------------------------------------------------------------------

def get_init_node_embed(graph, num_epochs, device):
    """Train PDW for num_epochs and return the frozen embedding table."""
    model = DeepWalkNeg(
        graph, embedding_dim=50, walk_length=3, r_hop=5, r_hop_size=5,
        walks_per_node=50, num_negative_samples=5, restart=0.15,
        sparse=True).to(device)

    loader = model.loader(batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train_epoch():
        model.train()
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for _ in range(num_epochs):
        train_epoch()

    return model().detach().cpu().clone()


class DeepWalkNeg(nn.Module):
    """
    Personalized DeepWalk with negative sampling.

    Embedding size is 2*embedding_dim + 1 to match the ToupleGNN input layout:
      [:embedding_dim]              → source (influence capacity) S_u
      [embedding_dim:2*embedding_dim] → target (tendency to be influenced) T_u
      [-1]                          → state scalar X_u
    """
    def __init__(self, graph, embedding_dim, walk_length, r_hop, r_hop_size,
                 walks_per_node=1, num_negative_samples=1, restart=0.5,
                 sparse=False):
        super().__init__()
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.walks_per_node = walks_per_node
        self.r_hop = r_hop
        self.r_hop_size = r_hop_size
        self.restart = restart
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(graph.num_nodes, embedding_dim * 2 + 1,
                                   sparse=sparse)
        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, batch=None):
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        return DataLoader(range(self.graph.num_nodes),
                          collate_fn=self.sample, **kwargs)

    def random_walk(self, start, walk_len, restart=0.5, rand=random.Random()):
        path = [start]
        for _ in range(walk_len):
            cur = path[-1]
            if self.graph.get_children(cur) and rand.random() >= restart:
                path.append(rand.choice(self.graph.get_children(cur)))
            else:
                path.append(path[0])
        return path

    def r_hop_neibors(self, start, r_hop):
        nodes_set = {start}
        queue = deque(nodes_set)
        for _ in range(r_hop):
            curr_nodes = set()
            while queue:
                curr = queue.popleft()
                curr_nodes.update(
                    c for c in self.graph.get_children(curr)
                    if c not in nodes_set)
            if not curr_nodes:
                break
            queue.extend(curr_nodes)
            nodes_set |= curr_nodes
        return list(nodes_set)

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)
        r_hop_n = {}
        walks = []
        for b in batch:
            s = b.item()
            rw = self.random_walk(s, self.walk_length, restart=self.restart)
            if s not in r_hop_n:
                r_hop_n[s] = self.r_hop_neibors(s, self.r_hop)
            rw.extend(random.choices(r_hop_n[s], k=self.r_hop_size))
            walks.append(rw)
        return torch.tensor(walks, dtype=torch.long)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        rw = torch.randint(self.graph.num_nodes,
                           (batch.size(0), self.walk_length + self.r_hop_size))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)
        return rw

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw, neg_rw):
        d = self.embedding_dim
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = self.embedding(start).view(pos_rw.size(0), 1, 2 * d + 1)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, 2 * d + 1)
        out = (h_start[:, :, -1]
               * (h_start[:, :, :d] * h_rest[:, :, d:2*d]).sum(-1)
               + h_rest[:, :, -1]).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start = self.embedding(start).view(neg_rw.size(0), 1, 2 * d + 1)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, 2 * d + 1)
        out = (h_start[:, :, -1]
               * (h_start[:, :, :d] * h_rest[:, :, d:2*d]).sum(-1)
               + h_rest[:, :, -1]).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')


class S2V_DUEL(nn.Module):
    """Dueling S2V baseline — unchanged from original."""
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):
        super(S2V_DUEL, self).__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim, True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)

        self.list_pre_pooling = []
        for _ in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)

        self.list_post_pooling = []
        for _ in range(self.len_post_pooling):
            post_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)

        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.fc_value = torch.nn.Linear(self.reg_hidden, 4 * self.reg_hidden)
            self.fc_adv = torch.nn.Linear(self.reg_hidden, 4 * self.reg_hidden)
            self.value = torch.nn.Linear(4 * self.reg_hidden, 1)
            self.adv = torch.nn.Linear(4 * self.reg_hidden, 1)
        else:
            self.fc_value = torch.nn.Linear(2 * embed_dim, 8 * embed_dim)
            self.fc_adv = torch.nn.Linear(2 * embed_dim, 8 * embed_dim)
            self.value = torch.nn.Linear(8 * embed_dim, 1)
            self.adv = torch.nn.Linear(8 * embed_dim, 1)

        for layer in [self.fc_value, self.fc_adv, self.value, self.adv]:
            torch.nn.init.normal_(layer.weight, mean=0, std=0.01)

    def forward(self, xv, adj):
        minibatch_size = xv.shape[0]
        num_node = xv.shape[1]
        mu = None
        for t in range(self.T):
            if t == 0:
                mu = torch.matmul(xv, self.mu_1).clamp(0)
            else:
                mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)
                mu_pool = torch.matmul(adj, mu)
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)
                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)
        q_1 = self.q_1(torch.matmul(xv.transpose(1, 2), mu)).expand(
            minibatch_size, num_node, self.embed_dim)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            value = self.fc_value(torch.mean(q_reg, dim=1, keepdim=True)).clamp(0)
            adv = self.fc_adv(q_reg).clamp(0)
            value = self.value(value)
            adv = self.adv(adv)
            adv_avg = torch.mean(adv, dim=1, keepdim=True)
            q = value + adv - adv_avg
        else:
            q_ = q_.clamp(0)
            value = self.fc_value(torch.mean(q_, dim=1, keepdim=True)).clamp(0)
            adv = self.fc_adv(q_).clamp(0)
            value = self.value(value)
            adv = self.adv(adv)
            adv_avg = torch.mean(adv, dim=1, keepdim=True)
            q = value + adv - adv_avg
        return q
