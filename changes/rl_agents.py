import random
import time
import os
from collections import namedtuple, deque
import numpy as np
import models
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max
from utils.graph_utils import COMM_FEAT_DIM

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)


class DQAgent:
    """Deep Q-learning agent for Influence Maximisation."""

    def __init__(self, args):
        self.model_name = args.model
        self.gamma = 0.99        # discount factor
        self.n_step = args.n_step

        self.training = not args.test
        self.T = args.T

        self.memory = ReplayMemory(args.memory_size)
        self.batch_size = args.bs

        self.double_dqn = args.double_dqn
        self.device = args.device

        self.node_dim = 2
        self.edge_dim = 4
        self.reg_hidden = args.reg_hidden
        self.embed_dim = args.embed_dim

        # num_communities is used only as an on/off flag (> 0 means "use
        # community features").  The actual feature dimension is always
        # COMM_FEAT_DIM=5 — fixed regardless of graph size.
        self.num_communities = getattr(args, 'num_communities', 0)

        # Cache of PDW node embeddings keyed by graph object id
        self.graph_node_embed = {}

        # ---- Model setup ----
        if self.model_name == 'S2V_DUEL':
            self.model = models.S2V_DUEL(
                reg_hidden=self.reg_hidden, embed_dim=self.embed_dim,
                node_dim=2, edge_dim=4, T=self.T, w_scale=0.01,
                avg=False).to(self.device)
            if self.training and self.double_dqn:
                self.target = models.S2V_DUEL(
                    reg_hidden=self.reg_hidden, embed_dim=self.embed_dim,
                    node_dim=2, edge_dim=4, T=self.T, w_scale=0.01,
                    avg=False).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'S2V_DQN':
            self.model = models.S2V_DQN(
                reg_hidden=self.reg_hidden, embed_dim=self.embed_dim,
                node_dim=2, edge_dim=4, T=self.T, w_scale=0.01, avg=False,
                num_communities=self.num_communities).to(self.device)
            if self.training and self.double_dqn:
                self.target = models.S2V_DQN(
                    reg_hidden=self.reg_hidden, embed_dim=self.embed_dim,
                    node_dim=2, edge_dim=4, T=self.T, w_scale=0.01, avg=False,
                    num_communities=self.num_communities).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'Tripling':
            self.model = models.Tripling(
                embed_dim=self.embed_dim, sgate_l1_dim=128, tgate_l1_dim=128,
                T=3, hidden_dims=[50, 50, 50], w_scale=0.01,
                num_communities=self.num_communities).to(self.device)
            if self.training and self.double_dqn:
                self.target = models.Tripling(
                    embed_dim=self.embed_dim, sgate_l1_dim=128,
                    tgate_l1_dim=128, T=3, hidden_dims=[50, 50, 50],
                    w_scale=0.01,
                    num_communities=self.num_communities).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            self.setup_graph_input = self.setup_graph_input_tripling

        else:
            raise NotImplementedError(f'RL Model {self.model_name}')

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if not self.training:
            cwd = os.getcwd()
            self.model.load_state_dict(
                torch.load(os.path.join(cwd, args.model_file)), strict=False)
            self.model.eval()
        elif self.training and getattr(args, 'resume', None):
            cwd = os.getcwd()
            checkpoint_path = os.path.join(cwd, args.resume)
            print(f'Resuming training from checkpoint: {checkpoint_path}')
            self.model.load_state_dict(
                torch.load(checkpoint_path), strict=False)
            if self.double_dqn:
                self.target.load_state_dict(self.model.state_dict())

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Community novelty flag helper
    # ------------------------------------------------------------------

    @staticmethod
    def _community_novelty_flags(graph, state_tensor):
        """
        Compute the dynamic community novelty flag for every node given the
        current seed set encoded in state_tensor.

        For each node v:
            flag[v] = log1p( |communities(v) − covered_communities| )

        where covered_communities = union of communities of all selected seeds.

        Interpretation:
            0.0  → all of v's communities are already covered by existing seeds,
                   or v has no community membership.
                   Selecting v will yield zero community-diversity bonus.
            0.69 → v has exactly 1 uncovered community (log1p(1)).
            1.39 → v has 3 uncovered communities (log1p(3)).

        Why log scale: prevents nodes in thousands of communities from
        producing huge values; keeps the feature in a similar range as
        the static structural features (cols 0-3).

        Why this is per-node and NOT the same for all nodes:
            Two nodes can both have uncovered communities, but one may be in
            3 new communities and the other in only 1.  After the GNN
            aggregates these flags over T=3 hops, a candidate whose reachable
            neighbourhood is dense with high-flag nodes gets a high aggregated
            source embedding — which is exactly the estimate of
            "community-spread potential" needed to learn the β*Δcomms reward.

        Parameters
        ----------
        graph : Graph
            The graph object (must have .communities, .nodes).
        state_tensor : 1-D torch.Tensor of shape (num_nodes,)
            state_tensor[i] == 1 if node i is a selected seed, else 0.

        Returns
        -------
        flags : float32 numpy array of shape (num_nodes,)
        """
        num_nodes = graph.num_nodes
        flags = np.zeros(num_nodes, dtype=np.float32)

        if not hasattr(graph, 'communities') or not graph.communities:
            return flags

        # Build the set of community IDs already reached by selected seeds
        selected_nodes = state_tensor.nonzero(as_tuple=True)[0].tolist()
        covered_comms = set()
        for sn in selected_nodes:
            covered_comms.update(graph.communities.get(sn, set()))

        # Per-node: how many of its communities are still NOT covered?
        for node in graph.nodes:
            node_comms = graph.communities.get(node, set())
            if node_comms:
                uncovered = node_comms - covered_comms
                flags[node] = np.log1p(len(uncovered))
            # nodes with no communities → flags[node] stays 0.0
            # nodes whose ALL communities are covered → also stays 0.0
            # This correctly signals: "selecting this node gives no β bonus"

        return flags

    def _attach_comm_features(self, graph, state_tensor):
        """
        Build the full (num_nodes, COMM_FEAT_DIM=5) community feature matrix
        for one graph at one step.

        Columns 0-3: static structural features (read from cache on the graph
                     object; computed once and stored on first call).
        Column 4:    dynamic novelty flag (recomputed from scratch every call
                     because it changes as seeds are added).

        The result is a float32 tensor of shape (num_nodes, 5).
        """
        # ---- Static features (columns 0-3) ----
        # get_community_features() is deterministic and graph-level, so we
        # cache it directly on the graph object to avoid recomputing it on
        # every single step.  The cache is invalidated if the graph changes
        # (shouldn't happen during a run).
        if not hasattr(graph, '_cached_comm_feats'):
            graph._cached_comm_feats = graph.get_community_features()
            # shape: (num_nodes, COMM_FEAT_DIM) with column 4 = 0.0

        comm_feats = torch.tensor(
            graph._cached_comm_feats.copy(),   # copy so we don't mutate cache
            dtype=torch.float32)

        # ---- Dynamic feature (column 4) ----
        # Recompute every call because the seed set changes each step.
        if self.num_communities > 0:
            novelty = self._community_novelty_flags(graph, state_tensor)
            comm_feats[:, 4] = torch.tensor(novelty, dtype=torch.float32)

        return comm_feats

    # ------------------------------------------------------------------
    # Graph input setup — Tripling model
    # ------------------------------------------------------------------

    def setup_graph_input_tripling(self, graphs, states, actions=None):
        """
        Build a batched PyG Data object for the Tripling model.

        x layout: [PDW_source (50-dim) | PDW_target (50-dim) | state (1-dim)]
        comm: community features with dynamic novelty flag in column 4
        """
        sample_size = len(graphs)
        data = []

        for i in range(sample_size):
            # Generate (or retrieve cached) PDW embedding for this graph
            if id(graphs[i]) not in self.graph_node_embed:
                self.graph_node_embed[id(graphs[i])] = \
                    models.get_init_node_embed(graphs[i], 30, self.device)

            with torch.no_grad():
                # Build node feature matrix: PDW embedding + current state bit
                x = self.graph_node_embed[id(graphs[i])].detach().clone()
                x = torch.cat(
                    (x, states[i].detach().clone().float().unsqueeze(1)),
                    dim=-1)

                edge_index = torch.tensor(
                    graphs[i].from_to_edges(),
                    dtype=torch.long).t().contiguous()
                edge_weight = torch.tensor(
                    [p[-1] for p in graphs[i].from_to_edges_weight()],
                    dtype=torch.float)

                y = actions[i].detach().clone() if actions is not None else None

                d = Data(x=x, edge_index=edge_index,
                         edge_weight=edge_weight, y=y)

                # Attach community features with fresh novelty flag
                if self.num_communities > 0:
                    d.comm = self._attach_comm_features(graphs[i], states[i])

                data.append(d)

        with torch.no_grad():
            loader = DataLoader(data, pin_memory=False, num_workers=4,
                                batch_size=sample_size, shuffle=False)
            for batch in loader:
                if actions is not None:
                    total_num = 0
                    for i in range(1, sample_size):
                        total_num += batch[i - 1].num_nodes
                        batch[i].y += total_num
                return batch.to(self.device)

    # ------------------------------------------------------------------
    # Graph input setup — S2V models
    # ------------------------------------------------------------------

    @torch.no_grad()
    def setup_graph_input_s2v(self, graphs, states, actions=None):
        """
        Build a batched PyG Data object for S2V_DQN / S2V_DUEL.

        Node features x: [1.0 (constant) | 1 - selected_flag]
        Edge features: [src_state | edge_prob | |src_state - dst_state| | 1.0]
        """
        sample_size = len(graphs)
        data = []

        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim)
            x[:, 1] = 1 - states[i].float()

            edge_index = torch.tensor(
                graphs[i].from_to_edges(),
                dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim)
            edge_attr[:, 1] = torch.tensor(
                [p[-1] for p in graphs[i].from_to_edges_weight()],
                dtype=torch.float)
            edge_attr[:, 0] = states[i].float()[edge_index[0]]
            edge_attr[:, 2] = torch.abs(
                states[i].float()[edge_index[0]] -
                states[i].float()[edge_index[1]])

            y = actions[i].clone() if actions is not None else None

            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            if self.num_communities > 0:
                d.comm = self._attach_comm_features(graphs[i], states[i])

            data.append(d)

        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            if actions is not None:
                total_num = 0
                for i in range(1, sample_size):
                    total_num += batch[i - 1].num_nodes
                    batch[i].y += total_num
            return batch.to(self.device)

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def setup_graph_pred(self, graphs, states, actions):
        """S2V variant that always expects actions (training mode)."""
        sample_size = len(states)
        data = []

        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim)
            x[:, 1] = 1 - states[i].float()

            edge_index = torch.tensor(
                graphs[i].from_to_edges(),
                dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim)
            edge_attr[:, 1] = torch.tensor(
                [p[-1] for p in graphs[i].from_to_edges_weight()],
                dtype=torch.float)
            edge_attr[:, 0] = states[i].float()[edge_index[0]]
            edge_attr[:, 2] = torch.abs(
                states[i].float()[edge_index[0]] -
                states[i].float()[edge_index[1]])

            y = actions[i].clone()

            d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            if self.num_communities > 0:
                d.comm = self._attach_comm_features(graphs[i], states[i])
            data.append(d)

        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            total_num = torch.tensor([0], dtype=torch.long)
            for i in range(1, sample_size):
                total_num.add_(batch[i - 1].num_nodes)
                batch[i].y = torch.add(batch[i].y, total_num)
            return batch.to(self.device)

    @torch.no_grad()
    def setup_graph_pred_all(self, graph, state):
        """S2V inference mode: Q on all nodes for a single graph."""
        x = torch.ones(graph.num_nodes, self.node_dim)
        x[:, 1] = 1 - state.float()

        edge_index = torch.tensor(
            graph.from_to_edges(), dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(graph.num_edges, self.edge_dim)
        edge_attr[:, 1] = torch.tensor(
            [p[-1] for p in graph.from_to_edges_weight()], dtype=torch.float)
        edge_attr[:, 0] = state.float()[edge_index[0]]
        edge_attr[:, 2] = torch.abs(
            state.float()[edge_index[0]] - state.float()[edge_index[1]])

        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if self.num_communities > 0:
            d.comm = self._attach_comm_features(graph, state)

        loader = DataLoader([d], batch_size=1, shuffle=False)
        for batch in loader:
            return batch.to(self.device)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, graph, state, epsilon, training=True, budget=None):
        """Select next seed node using ε-greedy policy."""

        if not training:
            # Greedy inference
            graph_input = self.setup_graph_input(
                [graph], state.unsqueeze(0))
            with torch.no_grad():
                q_a = self.model(graph_input)
            q_a[state.nonzero()] = -1e5

            if budget is None:
                return torch.argmax(q_a).detach().clone()
            else:
                return torch.topk(q_a.squeeze(1), budget)[1].detach().clone()

        # Training: ε-greedy
        available = (state == 0).nonzero().flatten().tolist()

        if epsilon > random.random():
            return torch.tensor([random.choice(available)],
                                 dtype=torch.long, device=self.device)
        else:
            graph_input = self.setup_graph_input([graph], state.unsqueeze(0))
            with torch.no_grad():
                q_a = self.model(graph_input)
            q_a[state.nonzero()] = -1e5

            available_tensor = torch.tensor(
                available, dtype=torch.long, device=self.device)
            max_position = (
                q_a == q_a[available_tensor].max().item()).nonzero()

            intersect = np.intersect1d(
                available_tensor.cpu().numpy(),
                max_position.cpu().contiguous().view(-1).numpy())
            if len(intersect) == 0:
                intersect = available_tensor.cpu().numpy()

            return torch.tensor([random.choice(intersect)],
                                 dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def memorize(self, env):
        """Store n-step transitions from the completed episode."""
        sum_rewards = [0.0]
        for reward in reversed(env.rewards):
            reward /= env.graph.num_nodes
            sum_rewards.append(reward + self.gamma * sum_rewards[-1])
        sum_rewards = sum_rewards[::-1]

        for i in range(len(env.states)):
            if i + self.n_step < len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long),
                    torch.tensor([env.actions[i]], dtype=torch.long),
                    torch.tensor(env.states[i + self.n_step], dtype=torch.long),
                    torch.tensor(
                        [sum_rewards[i]
                         - (self.gamma ** self.n_step)
                         * sum_rewards[i + self.n_step]],
                        dtype=torch.float),
                    env.graph)
            elif i + self.n_step == len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long),
                    torch.tensor([env.actions[i]], dtype=torch.long),
                    None,
                    torch.tensor([sum_rewards[i]], dtype=torch.float),
                    env.graph)

    def fit(self):
        """One gradient update step from the replay buffer."""
        if len(self.memory) == 0:
            return 0.0

        sample_size = min(self.batch_size, len(self.memory))
        transitions = self.memory.sample(sample_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            dtype=torch.bool, device=self.device)

        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_graphs = [
            batch.graph[i] for i, s in enumerate(batch.next_state)
            if s is not None]

        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        graph_batch = batch.graph

        # Current Q(s, a; Θ)
        state_action_values = self.model(
            self.setup_graph_input(
                graph_batch, state_batch, action_batch)
        ).squeeze(1)

        next_state_values = torch.zeros(sample_size, device=self.device)

        if len(non_final_next_states) > 0:
            batch_non_final = self.setup_graph_input(
                non_final_graphs, non_final_next_states)
            # Double DQN: target network provides next-state Q estimate
            if self.double_dqn:
                next_q = self.target(batch_non_final).squeeze(1)
            else:
                next_q = self.model(batch_non_final).squeeze(1)

            # Mask already-selected nodes so they can't be chosen
            next_q.add_(
                torch.cat(non_final_next_states).to(self.device) * (-1e5))
            next_state_values[non_final_mask] = scatter_max(
                next_q, batch_non_final.batch)[0].clamp_(min=0).detach()

        expected = (next_state_values * (self.gamma ** self.n_step)
                    + reward_batch.to(self.device))

        loss = self.criterion(state_action_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        if self.double_dqn:
            self.target.load_state_dict(self.model.state_dict())
            return True
        return False

    def save_model(self, file_name):
        torch.save(self.model.state_dict(),
                   os.path.join(os.getcwd(), file_name))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'graph'))


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


Agent = DQAgent
