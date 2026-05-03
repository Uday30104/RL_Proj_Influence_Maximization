import argparse
import sys
import os
import time
import torch
import utils.graph_utils as graph_utils
import rl_agents
import environment

def main():
    parser = argparse.ArgumentParser(description='Inference Script for INF-GNN-RL')
    parser.add_argument('--graph', type=str, required=True, help='Path to the large test graph file')
    parser.add_argument('--community_path', type=str, required=True, help='Path to community ground truth file for the test graph')
    parser.add_argument('--budget', type=int, default=5, help='Budget: number of seed nodes to select')
    parser.add_argument('--model', type=str, default='Tripling', help='Model architecture name (e.g., Tripling)')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the trained model file (.ckpt)')
    parser.add_argument('--num_communities', type=int, required=True, help='The number of communities the model was originally trained with (required to load the correct neural network shape)')
    parser.add_argument('--cpu', action='store_true', default=False, help='Force use CPU')
    
    args = parser.parse_args()

    # Set Device
    device = torch.device('cuda' if not(args.cpu) and torch.cuda.is_available() else 'cpu')
    args.device = device
    args.test = True # Enforce test mode so rl_agents automatically loads the weights

    # Load Graph
    print(f"Loading test graph from {args.graph}...")
    graph = graph_utils.read_graph(args.graph, ind=0, directed=True, community_path=args.community_path)
    
    # Ensure the graph object matches the model's expected community tensor shape
    graph.num_communities = args.num_communities
    graph.path_graph = args.graph
    args.graphs = [graph]

    # Required hyperparams for model initialization
    args.double_dqn = True
    args.T = 3
    args.memory_size = 50000
    args.reg_hidden = 32
    args.n_step = 2
    args.bs = 8
    args.lr = 1e-3
    
    if args.model == 'Tripling':
        args.embed_dim = 50
    else:
        args.embed_dim = 64

    # Load Agent
    print(f"Loading trained agent '{args.model}' from '{args.model_file}'...")
    # The agent checks args.test and automatically loads weights from args.model_file
    agent = rl_agents.Agent(args)

    # Load Environment for evaluation (Monte Carlo)
    print("Loading Monte Carlo environment for accurate evaluation...")
    # We use alpha=1.0 and beta=0.0 because for inference, we usually just want to measure 
    # true influence spread without reward scaling, but the metrics compute raw spread anyway.
    test_env = environment.Environment('IM', [graph], args.budget, method='MC', use_cache=True, alpha=1.0, beta=1.0)
    
    print(f"\nGenerating {args.budget} seed nodes...")
    start_time = time.time()
    
    test_env.reset(g_idx=0, training=False)
    state = torch.tensor(test_env.state, dtype=torch.long)
    
    # We use agent.select_action to generate all seeds in one forward pass
    seeds = agent.select_action(graph, state, epsilon=0.0, training=False, budget=args.budget).tolist()
    
    gen_time = time.time() - start_time
    
    print(f"Time taken to select seeds: {gen_time:.2f} seconds")
    
    print(f"\nEvaluating actual influence spread and communities reached via Monte Carlo simulation...")
    print(f"(Note: The environment automatically runs 10,000 independent cascade simulations and averages the results)")
    
    mc_start_time = time.time()
    _ = test_env.compute_reward(seeds)
    mc_time = time.time() - mc_start_time
    
    influence = test_env.prev_inf
    communities_reached = test_env.prev_comm
    
    print("\n" + "=" * 60)
    print("FINAL INFERENCE METRICS:")
    print("=" * 60)
    print(f"1. Selected Seed Nodes: {seeds}")
    print(f"2. Seed Generation Time (Model Inference): {gen_time:.4f} seconds")
    print(f"3. Evaluation Time (10,000 MC Simulations): {mc_time:.2f} seconds")
    print(f"4. Expected Total Influence Spread (Avg): {influence:.2f} nodes")
    print(f"5. Expected Unique Communities Reached (Avg): {communities_reached:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
