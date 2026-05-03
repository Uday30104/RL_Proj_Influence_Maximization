import numpy as np
import statistics
from multiprocessing import Pool
import random
import time
import utils.graph_utils as graph_utils

random.seed(123)
np.random.seed(123)


class Environment:
    """
    RL environment for Influence Maximisation with community-diversity reward.

    Reward formula (per step)
    -------------------------
        reward = alpha * (inf_gain / num_nodes)
                 + beta  * (comm_gain / num_communities)

    Both terms are normalised to [0, 1] so that alpha and beta have the same
    relative meaning on every graph regardless of size.

    Why normalise?
    --------------
    Training graphs: ~100 nodes, ~5 communities.
    Test graph:      potentially millions of nodes, thousands of communities.

    Without normalisation:
      inf_gain on training  ≈ 10–20 raw nodes per step
      inf_gain on test      ≈ 500–5000 raw nodes per step
    The model learns "an inf_gain of 15 is good" but that number means
    something completely different on the test graph.

    With normalisation by num_nodes:
      inf_gain / num_nodes on training ≈ 0.10–0.20
      inf_gain / num_nodes on test     ≈ 0.05–0.15  (comparable range)

    Similarly, without normalising comm_gain:
      On training: comm_gain of 1 out of 5 possible = 20% of budget
      On test:     comm_gain of 1 out of 3000 possible = 0.03% of budget
    The community bonus means something completely different at test time.

    With normalisation by num_communities:
      comm_gain / num_communities is always a fraction of total available.

    After normalisation, alpha and beta control relative importance cleanly:
      beta=1.0 → community diversity and influence spread weighted equally
      beta=2.0 → diversity twice as important as spread per unit
      beta=0.5 → spread slightly more important

    Recommended starting values: alpha=1.0, beta=2.0
    (community diversity 2× more important than proportional influence)
    """

    def __init__(self, name, graphs, budget, method='RR', use_cache=False,
                 training=True, alpha=1.0, beta=2.0):
        self.name   = name
        self.graphs = graphs
        self.budget = budget
        self.method = method
        self.alpha  = alpha
        self.beta   = beta

        self.use_cache = use_cache
        if self.use_cache:
            if self.method == 'MC':
                self.influences = {}
            elif self.method == 'RR':
                self.RRs_dict = {}
        self.training = training

    def reset_graphs(self, num_graphs=10):
        raise NotImplementedError()

    def reset(self, idx=None, training=True):
        if idx is None:
            self.graph = random.choice(self.graphs)
        else:
            self.graph = self.graphs[idx]

        self.state = [0] * self.graph.num_nodes

        # Running totals reset each episode
        self.prev_inf  = 0.0
        self.prev_comm = 0.0

        if self.use_cache and self.method == 'RR':
            self.RRs = self.RRs_dict.setdefault(id(self.graph), [])

        self.states  = []
        self.actions = []
        self.rewards = []
        self.training = training

    def compute_reward(self, S):
        num_process = 5
        num_trial   = 10000

        # ---- Get raw influence and community estimates ----
        need_compute = True
        if self.use_cache and self.method == 'MC':
            S_str = f"{id(self.graph)}.{','.join(map(str, sorted(S)))}"
            need_compute = S_str not in self.influences

        if need_compute:
            if self.method == 'MC':
                with Pool(num_process) as p:
                    results = p.map(
                        graph_utils.workerMC,
                        [[self.graph, S, int(num_trial / num_process)]
                         for _ in range(num_process)])
                es_inf   = statistics.mean([r[0] for r in results])
                es_comms = statistics.mean([r[1] for r in results])

            elif self.method == 'RR':
                if self.use_cache:
                    es_inf, es_comms = graph_utils.computeRR(
                        self.graph, S, num_trial, cache=self.RRs)
                else:
                    es_inf, es_comms = graph_utils.computeRR(
                        self.graph, S, num_trial)
            else:
                raise NotImplementedError(f'{self.method}')

            if self.use_cache and self.method == 'MC':
                self.influences[S_str] = (es_inf, es_comms)
        else:
            es_inf, es_comms = self.influences[S_str]

        # ---- Marginal gains ----
        inf_gain  = es_inf   - self.prev_inf
        comm_gain = es_comms - self.prev_comm
        self.prev_inf  = es_inf
        self.prev_comm = es_comms

        # ---- Normalise influence gain by graph size ----
        # Dividing by num_nodes converts raw node count into a fraction of
        # the graph that was newly activated.  This is always in [0, 1] and
        # means the same thing on a 100-node training graph and a million-node
        # test graph.
        norm_inf_gain = inf_gain / max(1, self.graph.num_nodes)

        # ---- Normalise community gain by number of communities ----
        # Dividing by num_communities converts "1 new community reached" into
        # a fraction of all available communities.  This prevents the community
        # signal from being trivially large on training graphs (5 communities)
        # and trivially small on the test graph (thousands of communities).
        #
        # Guard: if no community file was loaded num_communities == 0.
        # In that case we skip the community term entirely (it stays 0).
        num_comms = max(1, self.graph.num_communities)
        norm_comm_gain = comm_gain / num_comms

        # ---- Composite reward ----
        # Both terms are in [0, 1].
        # alpha controls how much raw spread matters.
        # beta  controls how much community diversity matters.
        # With alpha=1, beta=2: reaching 10% more of the graph (0.10) is
        # worth the same as covering 5% more of the community space (0.05 * 2).
        reward = self.alpha * norm_inf_gain + self.beta * norm_comm_gain

        self.rewards.append(reward)
        return reward

    def step(self, node, time_reward=None):
        if self.state[node] == 1:
            return

        self.states.append(self.state.copy())
        self.actions.append(node)
        self.state[node] = 1

        if self.name != 'IM':
            raise NotImplementedError(f'Environment {self.name}')

        S    = self.actions
        done = len(S) >= self.budget

        if self.training:
            reward = self.compute_reward(S)
        else:
            if done:
                if time_reward is not None:
                    start_time = time.time()
                reward = self.compute_reward(S)
                if time_reward is not None:
                    time_reward[0] = time.time() - start_time
            else:
                reward = None

        return (reward, done)
