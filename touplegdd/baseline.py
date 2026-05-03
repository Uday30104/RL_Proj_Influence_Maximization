import argparse
import time
import random
import utils.graph_utils as graph_utils


def main():
    parser = argparse.ArgumentParser(
        description='Baseline: evaluate a seed set on a graph via Monte Carlo.')

    parser.add_argument('--graph', type=str, required=True,
                        help='Path to graph file')
    parser.add_argument('--community_path', type=str, required=True,
                        help='Path to community file')
    parser.add_argument('--budget', type=int, default=5,
                        help='Number of seed nodes (used only for random mode)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Manually specify seed nodes e.g. --seeds 3 17 42 8 99')
    parser.add_argument('--num_trials', type=int, default=10000,
                        help='Number of Monte Carlo trials (default 10000)')
    parser.add_argument('--runs', type=int, default=1,
                        help='How many independent random seed sets to evaluate '
                             '(ignored if --seeds is provided)')

    args = parser.parse_args()

    # ---- Load graph ----
    print(f"Loading graph from {args.graph}...")
    graph = graph_utils.read_graph(
        args.graph, ind=0, directed=False, community_path=args.community_path)

    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Communities: {graph.num_communities}")

    # ---- Build seed sets to evaluate ----
    if args.seeds is not None:
        # Manual mode — validate the provided nodes
        invalid = [s for s in args.seeds if s < 0 or s >= graph.num_nodes]
        if invalid:
            print(f"\nERROR: These node IDs are out of range "
                  f"(valid range 0 to {graph.num_nodes - 1}): {invalid}")
            return
        seed_sets = [args.seeds]
        print(f"\nMode: MANUAL — evaluating provided seeds: {args.seeds}")
    else:
        # Random mode — sample `runs` independent seed sets
        seed_sets = [
            random.sample(range(graph.num_nodes), args.budget)
            for _ in range(args.runs)
        ]
        print(f"\nMode: RANDOM — evaluating {args.runs} random seed set(s) "
              f"of size {args.budget}")

    # ---- Evaluate each seed set ----
    print(f"Running {args.num_trials} Monte Carlo trials per seed set...\n")

    all_influences   = []
    all_communities  = []

    for i, seeds in enumerate(seed_sets):
        start = time.time()
        influence, communities = graph_utils.computeMC(
            graph, seeds, args.num_trials)
        elapsed = time.time() - start

        all_influences.append(influence)
        all_communities.append(communities)

        print(f"Run {i + 1}:")
        print(f"  Seeds selected      : {seeds}")
        print(f"  Influence spread    : {influence:.2f} nodes")
        print(f"  Communities reached : {communities:.2f}")
        print(f"  MC time             : {elapsed:.2f}s")
        print()

    # ---- Summary (only meaningful if runs > 1) ----
    if len(seed_sets) > 1:
        avg_inf  = sum(all_influences)  / len(all_influences)
        avg_comm = sum(all_communities) / len(all_communities)
        max_inf  = max(all_influences)
        max_comm = max(all_communities)
        print("=" * 50)
        print(f"SUMMARY over {len(seed_sets)} runs:")
        print(f"  Avg influence spread    : {avg_inf:.2f}")
        print(f"  Avg communities reached : {avg_comm:.2f}")
        print(f"  Best influence spread   : {max_inf:.2f}")
        print(f"  Best communities reached: {max_comm:.2f}")
        print("=" * 50)


if __name__ == '__main__':
    main()
