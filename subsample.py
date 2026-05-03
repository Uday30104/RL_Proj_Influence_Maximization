import networkx as nx
import os
import random
from collections import defaultdict, deque

# =============================================================================
# CONFIGURATION
# =============================================================================
GRAPH_FILE = 'actual_graph.txt'
COMM_FILE  = 'top_5000_communities.txt'

OUTPUT_GRAPH_DIR = 'train_graphs'
OUTPUT_COMM_DIR  = 'train_comms'
TEST_GRAPH_FILE  = 'test_graph.txt'
TEST_COMM_FILE   = 'test_comms.txt'

NUM_TRAIN_GRAPHS = 70
SUBGRAPH_SIZE    = 100
TRAIN_RATIO      = 0.80

# Jaccard overlap: two communities with overlap above this are considered
# near-duplicates; the smaller one is discarded.
MAX_JACCARD_OVERLAP = 0.6

# How many structurally-distant anchor communities to seed the concurrent BFS.
NUM_ANCHORS = 4

# Community count bounds per subgraph AFTER Jaccard filtering.
MIN_DISTINCT_COMMS = 4
MAX_DISTINCT_COMMS = 8

# Minimum Jaccard DISTANCE between anchor communities.
# 0.9 means anchors share at most 10% of their nodes.
MIN_ANCHOR_DISTANCE = 0.9

# =============================================================================


# =============================================================================
# FIX 1 — Safe edge writer
# nx.write_edgelist writes isolated nodes as bare lines ("42\n") which breaks
# read_graph (expects exactly two tokens per line).  This writer only writes
# edges and formats them as plain "u v" with no metadata.
# =============================================================================

def write_edgelist(G, path):
    """
    Write graph edges as plain 'src dst' pairs, one per line.
    Only edges are written — isolated nodes are skipped because read_graph
    builds the node set from edges only, and isolated nodes contribute
    nothing to influence spread.
    Each undirected edge is written once; read_graph with directed=False
    automatically adds both directions.
    """
    with open(path, 'w') as f:
        for u, v in G.edges():
            f.write(f'{int(u)} {int(v)}\n')


# =============================================================================

def load_data():
    print(f"Loading graph from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
        G = nx.read_edgelist(f, delimiter='\t', nodetype=int)

    print(f"Loading communities from {COMM_FILE}...")
    node_to_comms = defaultdict(set)
    comm_to_nodes = defaultdict(set)

    with open(COMM_FILE, 'r', encoding='utf-8') as f:
        for comm_id, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            nodes = list(map(int, line.split('\t')))
            for n in nodes:
                node_to_comms[n].add(comm_id)
                comm_to_nodes[comm_id].add(n)

    print(f"  Loaded {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, "
          f"{len(comm_to_nodes)} communities.")
    return G, node_to_comms, comm_to_nodes


def macro_split(G, train_ratio):
    """
    Snowball BFS macro-split: carve the graph into a train region and a
    test region.  The test region is saved as the single large test graph.
    """
    print("\nExecuting snowball macro-split...")
    target_train = int(G.number_of_nodes() * train_ratio)

    train_nodes = set()
    visited     = set()
    start       = random.choice(list(G.nodes()))
    queue       = deque([start])
    visited.add(start)

    while queue and len(train_nodes) < target_train:
        curr = queue.popleft()
        train_nodes.add(curr)
        for nb in G.neighbors(curr):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    G_train = G.subgraph(train_nodes).copy()
    G_test  = G.subgraph(set(G.nodes()) - train_nodes).copy()

    print(f"  Train region: {G_train.number_of_nodes()} nodes, "
          f"{G_train.number_of_edges()} edges")
    print(f"  Test  region: {G_test.number_of_nodes()} nodes, "
          f"{G_test.number_of_edges()} edges")
    return G_train, G_test


def save_test_data(G_test, node_to_comms):
    """
    Relabel test graph nodes to 0..N-1 and save graph + communities.
    Uses write_edgelist (not nx.write_edgelist) to avoid isolated-node lines.
    """
    print("\nSaving test data...")

    # Relabel to contiguous 0..N-1
    mapping = {old: new for new, old in enumerate(G_test.nodes())}
    G_test_r = nx.relabel_nodes(G_test, mapping)

    # Save graph — plain "u v" edges only
    write_edgelist(G_test_r, TEST_GRAPH_FILE)

    # Save communities using relabelled node IDs
    test_comms = defaultdict(list)
    for old, new in mapping.items():
        for c in node_to_comms.get(old, []):
            test_comms[c].append(new)

    with open(TEST_COMM_FILE, 'w', encoding='utf-8') as f:
        for members in test_comms.values():
            if len(members) >= 2:
                f.write('\t'.join(map(str, sorted(members))) + '\n')

    print(f"  Saved: {TEST_GRAPH_FILE}  ({G_test_r.number_of_nodes()} nodes, "
          f"{G_test_r.number_of_edges()} edges)")
    print(f"  Saved: {TEST_COMM_FILE}  ({len(test_comms)} communities)")


# =============================================================================
# Pick structurally distant anchor communities
# =============================================================================

def pick_diverse_anchors(comm_to_nodes, train_nodes_set,
                         k=NUM_ANCHORS,
                         min_size=5, max_size=50,
                         min_distance=MIN_ANCHOR_DISTANCE):
    """
    Greedily select k communities whose node sets are as disjoint as possible.

    Single-anchor BFS creates a subgraph centred on one community where
    high-degree nodes dominate influence AND cover most communities naturally
    — no trade-off exists for the agent to learn from.

    Starting from k distant communities forces the subgraph to genuinely span
    different network regions so the agent must choose between
    high-influence-in-covered-community vs lower-influence-in-novel-community.

    Returns list of (comm_id, frozenset_of_nodes) or None if selection fails.
    """
    candidates = []
    for c_id, members in comm_to_nodes.items():
        local = members.intersection(train_nodes_set)
        if min_size <= len(local) <= max_size:
            candidates.append((c_id, frozenset(local)))

    if len(candidates) < k:
        return None

    random.shuffle(candidates)
    selected = [candidates[0]]

    for _ in range(k - 1):
        best, best_dist = None, -1.0

        for cid, members in candidates:
            if any(cid == s[0] for s in selected):
                continue

            # Minimum Jaccard distance to all currently selected anchors
            min_dist = min(
                1.0 - len(members & s_members) / max(1, len(members | s_members))
                for _, s_members in selected
            )

            if min_dist > best_dist:
                best_dist = min_dist
                best = (cid, members)

        if best is None or best_dist < min_distance:
            break

        selected.append(best)

    return selected if len(selected) >= 2 else None


# =============================================================================
# Concurrent BFS from multiple anchors
# =============================================================================

def concurrent_bfs(G_train, anchor_communities, target_size):
    """
    Grow a subgraph simultaneously from multiple anchor communities.

    Standard single-source BFS produces a ball dense at the centre and thin
    at the edges — one dominant community, others as sparse fringe nodes.
    Concurrent BFS grows k balls in parallel so the subgraph genuinely spans
    all community regions with comparable density, forcing real trade-offs.

    Each frontier expands one node per round in round-robin order.
    Nodes are claimed by the first frontier to reach them.
    """
    start_nodes = [random.choice(list(members))
                   for _, members in anchor_communities]

    sub_nodes = set(start_nodes)
    frontiers = [deque([s]) for s in start_nodes]

    while len(sub_nodes) < target_size:
        any_progress = False

        for frontier in frontiers:
            if not frontier:
                continue

            curr = frontier.popleft()
            neighbors = list(G_train.neighbors(curr))
            random.shuffle(neighbors)

            for nb in neighbors:
                if nb not in sub_nodes:
                    sub_nodes.add(nb)
                    frontier.append(nb)
                    any_progress = True
                    if len(sub_nodes) >= target_size:
                        break

            if len(sub_nodes) >= target_size:
                break

        if not any_progress:
            break

    return sub_nodes


# =============================================================================
# FIX 2 — Jaccard filter keeping sets internally
# Original bug: stored members as list, then tried set intersection (&) on them
# Fix: keep as sets throughout, convert to list only at return
# =============================================================================

def filter_distinct_communities(local_comms_dict, max_jaccard):
    """
    Remove near-duplicate communities.
    Two communities with Jaccard similarity > max_jaccard are duplicates;
    the smaller one is discarded.  Only communities with >= 2 members kept.
    """
    # Build candidates as sets (required for & and | operations below)
    candidates = {cid: set(m) for cid, m in local_comms_dict.items()
                  if len(m) >= 2}

    # Process largest communities first — keep the more representative one
    sorted_cands = sorted(candidates.items(),
                          key=lambda x: len(x[1]), reverse=True)

    distinct = {}  # stores sets internally throughout

    for cid, members in sorted_cands:
        is_distinct = True
        for d_members in distinct.values():
            inter = len(members & d_members)   # set & set — no TypeError
            union = len(members | d_members)
            if union > 0 and inter / union > max_jaccard:
                is_distinct = False
                break
        if is_distinct:
            distinct[cid] = members            # keep as set

    # Convert to lists only here, at the boundary where the caller needs them
    return {cid: list(members) for cid, members in distinct.items()}


# =============================================================================
# Main extraction loop
# =============================================================================

def extract_subgraphs(G_train, node_to_comms, comm_to_nodes):
    """
    Extract NUM_TRAIN_GRAPHS subgraphs, each:
      - exactly SUBGRAPH_SIZE nodes
      - a single connected component
      - between MIN_DISTINCT_COMMS and MAX_DISTINCT_COMMS distinct communities
      - grown from multiple distant anchors so influence/diversity are in tension
    """
    print(f"\nExtracting {NUM_TRAIN_GRAPHS} subgraphs "
          f"(anchors={NUM_ANCHORS}, "
          f"comms=[{MIN_DISTINCT_COMMS},{MAX_DISTINCT_COMMS}], "
          f"size={SUBGRAPH_SIZE})...")

    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_COMM_DIR,  exist_ok=True)

    train_nodes_set = set(G_train.nodes())
    generated = 0
    attempts  = 0

    while generated < NUM_TRAIN_GRAPHS:
        attempts += 1

        # --- Step 1: Pick k structurally distant anchor communities ---
        anchors = pick_diverse_anchors(comm_to_nodes, train_nodes_set)
        if anchors is None:
            continue

        # --- Step 2: Grow subgraph concurrently from all anchors ---
        sub_nodes = concurrent_bfs(G_train, anchors, SUBGRAPH_SIZE)

        if len(sub_nodes) < SUBGRAPH_SIZE:
            continue

        # --- Step 3: Build subgraph ---
        G_sub = G_train.subgraph(sub_nodes).copy()

        # FIX 3 — Connectivity check
        # concurrent_bfs can occasionally produce a disconnected subgraph if
        # the anchors are in separate components of G_train.  We keep only the
        # largest connected component.  If it is too small, retry entirely.
        if not nx.is_connected(G_sub):
            largest_cc = max(nx.connected_components(G_sub), key=len)
            if len(largest_cc) < SUBGRAPH_SIZE * 0.8:
                # Lost too many nodes — not worth keeping
                continue
            G_sub = G_sub.subgraph(largest_cc).copy()

        # --- Step 4: Relabel nodes to contiguous 0..N-1 ---
        # This is essential — DeepWalkNeg creates an embedding table of size
        # graph.num_nodes = len(nodes).  If node IDs have gaps the embedding
        # lookup crashes with an out-of-bounds CUDA error.
        mapping = {old: new for new, old in enumerate(G_sub.nodes())}
        G_sub_r = nx.relabel_nodes(G_sub, mapping)

        # --- Step 5: Collect and remap communities ---
        raw_comms = defaultdict(list)
        for old, new in mapping.items():
            for c_id in node_to_comms.get(old, []):
                raw_comms[c_id].append(new)

        # --- Step 6: Remove near-duplicate communities ---
        distinct_comms = filter_distinct_communities(raw_comms,
                                                     MAX_JACCARD_OVERLAP)

        # --- Step 7: Enforce community count bounds ---
        n_comms = len(distinct_comms)
        if not (MIN_DISTINCT_COMMS <= n_comms <= MAX_DISTINCT_COMMS):
            continue

        # --- Step 8: Save using safe edge writer ---
        graph_path = os.path.join(OUTPUT_GRAPH_DIR,
                                  f'train_graph_{generated:03d}.txt')
        comm_path  = os.path.join(OUTPUT_COMM_DIR,
                                  f'train_comm_{generated:03d}.txt')

        # write_edgelist writes plain "u v" lines only — no isolated nodes,
        # no metadata, no tab-separated extras that confuse read_graph
        write_edgelist(G_sub_r, graph_path)

        with open(comm_path, 'w', encoding='utf-8') as f:
            for members in distinct_comms.values():
                f.write('\t'.join(map(str, sorted(members))) + '\n')

        generated += 1
        if generated % 10 == 0:
            print(f"  {generated}/{NUM_TRAIN_GRAPHS} done "
                  f"(attempts: {attempts}, "
                  f"last: {G_sub_r.number_of_nodes()} nodes, "
                  f"{n_comms} communities)")

    print(f"\nFinished. {generated} subgraphs saved in {attempts} attempts.")
    print(f"Graphs → {OUTPUT_GRAPH_DIR}/")
    print(f"Comms  → {OUTPUT_COMM_DIR}/")


# =============================================================================

if __name__ == '__main__':
    random.seed(42)
    G, node_to_comms, comm_to_nodes = load_data()
    G_train, G_test = macro_split(G, TRAIN_RATIO)
    save_test_data(G_test, node_to_comms)
    extract_subgraphs(G_train, node_to_comms, comm_to_nodes)
    print("\nAll tasks completed.")