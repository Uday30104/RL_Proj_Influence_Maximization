import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool

random.seed(123)
np.random.seed(123)

# Fixed community feature dimension — always 5, on every graph, regardless of
# how many communities exist.  The 5 features are structural statistics that
# have identical meaning on a 12-community training graph and a 3191-community
# test graph.  See get_community_features() for the exact definitions.
#
# Column layout:
#   0  log1p(number of communities this node belongs to)
#   1  mean size of this node's communities / graph size
#   2  fraction of this node's neighbours that share at least one community
#   3  size of this node's largest community / graph size
#   4  DYNAMIC — log1p(uncovered communities at this node given current seeds)
#               filled to 0.0 here; overwritten each step in rl_agents.py
COMM_FEAT_DIM = 5


class Graph:
    ''' graph class '''
    def __init__(self, nodes, edges, children, parents, communities=None, num_communities=0):
        self.nodes = nodes                  # set()
        self.edges = edges                  # dict{(src,dst): weight}
        self.children = children            # dict{node: set()}
        self.parents = parents              # dict{node: set()}

        self.communities = communities if communities is not None else {}
        # dict{node: set(comm_ids)} — a node can belong to zero, one, or many communities
        self.num_communities = num_communities

        # convert neighbour sets to sorted lists for deterministic iteration
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def get_children(self, node):
        return self.children.get(node, [])

    def get_parents(self, node):
        return self.parents.get(node, [])

    def get_prob(self, edge):
        return self.edges[edge]

    def get_adj(self):
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge]
            self._adj = csr_matrix(self._adj)
        return self._adj

    def from_to_edges(self):
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight

    # ------------------------------------------------------------------
    # Community feature extraction — fixed dimension, graph-independent
    # ------------------------------------------------------------------
    def get_community_features(self):
        """
        Returns a float32 numpy array of shape (num_nodes, COMM_FEAT_DIM=5).

        Columns 0-3 are STATIC structural statistics computed once per graph.
        Column 4 is a DYNAMIC slot initialised to 0.0 here; it is overwritten
        every step in rl_agents.setup_graph_input_* to encode how much novel
        community coverage this node's cascade could still contribute given
        the seeds already chosen.

        Because every feature is a ratio or a log-count that does not depend
        on the total number of communities, the same 5-dimensional vector is
        meaningful on any graph — a 12-community training graph and a
        3191-community test graph produce identically shaped tensors, and the
        learned projection weights transfer without any shape mismatch.

        Nodes not present in the community file get all-zero rows, which the
        model learns to interpret as "no community signal available".
        """
        # num_nodes is len(self.nodes).  We assume node IDs are 0-indexed and
        # contiguous (guaranteed by read_graph after the `ind` offset), so
        # array index == node ID throughout.
        features = np.zeros((self.num_nodes, COMM_FEAT_DIM), dtype=np.float32)

        if not self.communities or self.num_communities == 0:
            # No community file was loaded.  Return all-zeros so the model
            # gets a consistent zero input rather than crashing.
            return features

        # --- Precompute community sizes (done once per graph call) ---
        # Each node in `communities` may belong to several community IDs.
        # We tally how many nodes each community contains.
        comm_sizes = {}
        for node, comms in self.communities.items():
            for c in comms:
                comm_sizes[c] = comm_sizes.get(c, 0) + 1

        # --- Per-node feature computation ---
        for node in self.nodes:
            node_comms = self.communities.get(node, set())
            # .get with empty-set default handles nodes absent from the
            # community file (zero-community case) without KeyError.

            # Column 0: log-scaled community membership count
            #   0 communities → log1p(0) = 0.0
            #   1 community   → log1p(1) ≈ 0.69
            #   5 communities → log1p(5) ≈ 1.79
            # Log scale prevents nodes in thousands of overlapping communities
            # from producing huge values that destabilise the projection.
            features[node, 0] = np.log1p(len(node_comms))

            if len(node_comms) == 0:
                # Columns 1-3 stay 0; column 4 filled dynamically elsewhere.
                continue

            # sizes: list of membership counts for each community this node
            # is in.  E.g. node in {comm_0 (size 4), comm_2 (size 3)} → [4,3]
            sizes = [comm_sizes.get(c, 1) for c in node_comms]

            # Column 1: mean community size relative to graph size
            #   Approaches 0 for nodes in tiny niche clusters.
            #   Approaches 1 for nodes inside the single dominant community.
            #   Always in (0, 1] because every community has ≥1 member.
            features[node, 1] = float(np.mean(sizes)) / self.num_nodes

            # Column 2: intra-community edge fraction
            #   For each outgoing neighbour, check whether it shares at
            #   least one community with the current node.
            #   0.0 → node sits on the pure boundary between communities
            #         (its cascade immediately crosses into other communities)
            #   1.0 → node is deep inside a single community
            #         (its cascade stays within the same community)
            # This is the key structural signal for the GNN to estimate
            # how much community diversity a cascade from this node produces.
            neighbours = self.children.get(node, [])
            if len(neighbours) > 0:
                intra = sum(
                    1 for nb in neighbours
                    if node_comms & self.communities.get(nb, set())
                )
                features[node, 2] = intra / len(neighbours)
            # isolated node: stays 0.0 (correct — no edges means no spread)

            # Column 3: largest community size relative to graph size
            #   Identifies whether the node is part of the dominant community.
            #   High value → likely already covered by an earlier seed.
            features[node, 3] = float(max(sizes)) / self.num_nodes

            # Column 4: stays 0.0 — filled dynamically in rl_agents.py
            # based on which communities the current seed set has already
            # reached.  See _community_novelty_flags() in rl_agents.py.

        return features


def read_graph(path, ind=0, directed=False, community_path=None):
    """
    Load an edge-list graph file.

    Edge file format: one edge per line, "src dst" (space-separated).
    Lines starting with # or % are treated as comments.

    Community file format (SNAP ground-truth style):
    One community per line; each line is a space-separated list of node IDs
    that belong to that community.  A node can appear on multiple lines
    (multiple community membership).  Nodes absent from the file get an empty
    community set.

    ind: integer offset subtracted from every node ID in the file.
         Use ind=1 for 1-indexed files so that internal IDs start at 0.
    directed: if False, every edge (u,v) is stored as both (u,v) and (v,u).
    """
    parents = {}
    children = {}
    edges = {}
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split()
            src = int(row[0]) - ind
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst)
            parents.setdefault(dst, set()).add(src)
            edges[(src, dst)] = 0.0
            if not directed:
                children.setdefault(dst, set()).add(src)
                parents.setdefault(src, set()).add(dst)
                edges[(dst, src)] = 0.0

    # Set each edge probability to 1 / in-degree of the destination node.
    # This is the standard "in-degree" setting used throughout the paper.
    for src, dst in edges:
        edges[(src, dst)] = 1.0 / len(parents[dst])

    # --- Parse community file ---
    communities = {}
    num_communities = 0
    if community_path:
        with open(community_path, 'r') as f:
            for comm_id, line in enumerate(f):
                line = line.strip()
                if not len(line) or line.startswith('#') or line.startswith('%'):
                    continue
                # Each line: space-separated node IDs in this community
                for node_str in line.split():
                    node = int(node_str) - ind
                    if node in nodes:
                        # setdefault creates the set if missing; .add handles
                        # the case where the node is already in other communities
                        communities.setdefault(node, set()).add(comm_id)
                        num_communities = max(num_communities, comm_id + 1)

    return Graph(nodes, edges, children, parents, communities, num_communities)


# ----------------------------------------------------------------------
# Influence estimation: Monte Carlo
# ----------------------------------------------------------------------

def computeMC(graph, S, R):
    """
    Estimate expected influence spread and unique communities reached under
    the IC model using R Monte Carlo trials.

    Returns (expected_nodes_activated, expected_unique_communities_activated).
    Both are averages over R independent cascade simulations.
    """
    sources = set(S)
    total_inf = 0
    total_comms = 0

    for _ in range(R):
        # Run one IC cascade from S
        activated = sources.copy()
        queue = deque(activated)
        while True:
            newly_activated = set()
            while queue:
                curr = queue.popleft()
                for child in graph.get_children(curr):
                    if child not in activated and random.random() <= graph.edges[(curr, child)]:
                        newly_activated.add(child)
            if not newly_activated:
                break
            queue.extend(newly_activated)
            activated |= newly_activated

        total_inf += len(activated)

        # Count unique communities touched by the full activated set
        trial_comms = set()
        for node in activated:
            trial_comms.update(graph.communities.get(node, set()))
        total_comms += len(trial_comms)

    return total_inf / R, total_comms / R


def workerMC(x):
    """Multiprocessing wrapper for computeMC."""
    return computeMC(x[0], x[1], x[2])


# ----------------------------------------------------------------------
# Influence estimation: Reverse Reachability (RR)
# ----------------------------------------------------------------------

def computeRR(graph, S, R, cache=None):
    """
    Estimate expected influence spread and unique communities reached under
    the IC model using Reverse Reachability sets.

    Each RR set is stored as (target_node, reverse_reachable_set).
    If cache is provided and already populated, reuses it (fast path).
    If cache is provided and empty, generates R new sets and fills it.

    Returns (estimated_influence, estimated_unique_communities).

    How community estimation works:
      A seed set S "covers" an RR set if at least one seed is in the reverse
      reachable set — meaning S could have activated target_node.
      We count unique community IDs of all covered target nodes; these are the
      communities the spread "reaches" on average.
    """
    covered = 0
    generate_RR = False

    if cache is not None:
        if len(cache) > 0:
            # Fast path: cache already populated, just count coverage
            covered_targets = []
            for target, RR in cache:
                if any(s in RR for s in S):
                    covered += 1
                    covered_targets.append(target)

            unique_comms = set()
            for t in covered_targets:
                unique_comms.update(graph.communities.get(t, set()))

            return (covered * 1.0 / R * graph.num_nodes,
                    len(unique_comms))
        else:
            generate_RR = True

    # Generate R new RR sets
    unique_comms = set()
    for _ in range(R):
        target = random.randint(0, graph.num_nodes - 1)
        # Build the reverse reachable set from target by walking edges backwards
        source_set = {target}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while queue:
                curr = queue.popleft()
                for parent in graph.get_parents(curr):
                    if parent not in source_set and random.random() <= graph.edges[(parent, curr)]:
                        curr_source_set.add(parent)
            if not curr_source_set:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set

        if any(s in source_set for s in S):
            covered += 1
            # target is activatable by S; count its communities
            unique_comms.update(graph.communities.get(target, set()))

        if generate_RR:
            cache.append((target, source_set))

    return (covered * 1.0 / R * graph.num_nodes,
            len(unique_comms))


def workerRR(x):
    """Multiprocessing wrapper for computeRR."""
    return computeRR(x[0], x[1], x[2])


def computeRR_inc(graph, S, R, cache=None, l_c=None):
    """
    Incremental RR computation (influence only, no community tracking).
    Used internally; kept for backward compatibility.
    """
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True

    for _ in range(R):
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while queue:
                curr = queue.popleft()
                for parent in graph.get_parents(curr):
                    if parent not in source_set and random.random() <= graph.edges[(parent, curr)]:
                        curr_source_set.add(parent)
            if not curr_source_set:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set

        if any(s in source_set for s in S):
            covered += 1
        if generate_RR:
            cache.append(source_set)

    return covered * 1.0 / R * graph.num_nodes


if __name__ == '__main__':
    path = "../soc-dolphins.txt"
    num_trial = 10000

    graph = read_graph(path, ind=1, directed=False)

    print('Generating seed sets:')
    list_S = []
    for _ in range(10):
        list_S.append(random.sample(range(graph.num_nodes), k=random.randint(3, 10)))
        print(f'({str(list_S[-1])[1:-1]})')

    print('Cached single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    RR_cache = []
    for S in list_S:
        time_start = time.time()
        es_infs.append(computeRR(graph, S, num_trial, cache=RR_cache))
        times.append(time.time() - time_start)
    time_2 = time.time()

    for i in range(10):
        print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f}s; inf={es_infs[i][0]:.1f} comms={es_infs[i][1]}')
    print(f'Total gross time: {time_2 - time_1:.2f}s')
    print(f'Total time: {sum(times):.2f}s')
