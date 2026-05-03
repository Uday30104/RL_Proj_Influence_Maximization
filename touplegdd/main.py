import argparse
import sys
import os
import re
import time
import datetime
import numpy as np
import torch
import utils.graph_utils as graph_utils
import rl_agents
import environment
import runner
import logging

torch.manual_seed(123)
np.random.seed(123)

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='INF-GNN-RL')
parser.add_argument('--budget', type=int, default=6,
                    help='budget: number of seed nodes to select')
parser.add_argument('--graph', type=str, metavar='GRAPH_PATH',
                    default='soc-dolphins.txt',
                    help='path to graph file or directory of graph files')
parser.add_argument('--agent', type=str, metavar='AGENT_CLASS',
                    default='Agent')
parser.add_argument('--model', type=str, default='Tripling',
                    help='model class: Tripling | S2V_DQN | S2V_DUEL')
parser.add_argument('--model_file', type=str, default='tripling.ckpt')
parser.add_argument('--epoch', type=int, metavar='nepoch', default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--bs', type=int, default=8, help='minibatch size')
parser.add_argument('--n_step', type=int, default=2,
                    help='n-step return horizon')
parser.add_argument('--cpu', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str,
                    default='IM')

# Community & reward arguments
parser.add_argument('--community_path', type=str, default=None,
                    help='path to community ground-truth file (or directory)')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='reward weight for raw influence gain')
parser.add_argument('--beta', type=float, default=10.0,
                    help='reward weight for newly reached communities')

# Resume / checkpoint
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path to resume training from')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='starting epoch for epsilon schedule when resuming')


def main():
    args = parser.parse_args()
    logging.info('Loading graph %s' % args.graph)

    device = torch.device(
        'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')
    args.device = device

    # ---- Load graph(s) ----
    natural_key = lambda s: [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r'(\d+)', s)]

    path_graphs = []
    path_comms = []

    if os.path.isdir(args.graph):
        graph_files = sorted(
            [f for f in os.listdir(args.graph) if not f.startswith('.')],
            key=natural_key)
        path_graphs = [os.path.join(args.graph, f) for f in graph_files]

        if args.community_path and os.path.isdir(args.community_path):
            comm_files = sorted(
                [f for f in os.listdir(args.community_path)
                 if not f.startswith('.')],
                key=natural_key)
            if len(comm_files) != len(graph_files):
                logging.warning(
                    'Number of graph files and community files do not match!')
            path_comms = [os.path.join(args.community_path, f)
                          for f in comm_files]
        else:
            path_comms = [args.community_path] * len(path_graphs)
    else:
        path_graphs = [args.graph]
        path_comms = [args.community_path]

    graph_lst = []
    for pg, pc in zip(path_graphs, path_comms):
        graph_lst.append(
            graph_utils.read_graph(pg, ind=0, directed=True,
                                   community_path=pc))

    for i, pg in enumerate(path_graphs):
        graph_lst[i].path_graph = pg

    # ---- Community feature flag ----
    # args.num_communities is used ONLY as an on/off switch (> 0 means
    # "build and use the community projection layers").
    #
    # The actual neural-network input dimension is always COMM_FEAT_DIM=5
    # (imported from graph_utils), regardless of how many communities any
    # graph has.  We do NOT pad or align graphs to a shared community count
    # any more — that was the source of the train/test shape mismatch.
    #
    # Log the max community count for reference, but don't use it for
    # model sizing.
    max_comms = max(g.num_communities for g in graph_lst) if graph_lst else 0
    has_communities = max_comms > 0
    args.num_communities = 1 if has_communities else 0

    if has_communities:
        logging.info(
            f'Community features enabled.  '
            f'Max communities across loaded graphs: {max_comms}.  '
            f'GNN input dimension: COMM_FEAT_DIM=5 (fixed).')
    else:
        logging.info('No community file provided — community features disabled.')

    args.graphs = graph_lst
    args.double_dqn = True

    if not args.test:
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        os.makedirs(time_stamp, exist_ok=True)
        args.model_file = os.path.join(time_stamp, args.model_file)

    args.T = 3
    args.memory_size = 50000
    args.reg_hidden = 32
    args.embed_dim = 50 if args.model == 'Tripling' else 64

    # ---- Agent ----
    logging.info(f'Loading agent {args.model}')
    agent = rl_agents.Agent(args)

    # ---- Environments ----
    logging.info('Loading environment %s' % args.environment_name)
    train_env = environment.Environment(
        args.environment_name, graph_lst, args.budget,
        method='RR', use_cache=True, alpha=args.alpha, beta=args.beta)
    test_env = environment.Environment(
        args.environment_name, graph_lst, args.budget,
        method='MC', use_cache=True, alpha=args.alpha, beta=args.beta)

    # ---- Run ----
    print('Running simulation')
    my_runner = runner.Runner(train_env, test_env, agent, not args.test)
    if not args.test:
        my_runner.train(args.epoch, args.model_file,
                        'list_cumul_reward.txt',
                        start_epoch=args.start_epoch,
                        resume=args.resume is not None)
    else:
        my_runner.test(num_trials=10)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Total time: {time.time() - start_time:.2f}s')
