# ToupleGDD

Implementation of "ToupleGDD: A Fine-Designed Solution of Influence Maximization by Deep Reinforcement Learning" (https://arxiv.org/abs/2210.07500)

Run the code
------------
 python main.py   --graph train_graphs_new/   --community_path train_comms_new/   --budget 5   --alpha 10.0   --beta 1.0   --bs 16   --epoch 10000   --model Tripling   --model_file tripling.ckpt   --n_step 2



 inference: 

 python inference.py   --graph test_graph_new.txt   --community_path test_comms_new.txt   --model_file "2026-05-03_14:14:18/tripling.ckpt650"   --num_communities 1   --budget 5


#### Train ToupleGDD model

	python main.py --graph train_data \
                     --model Tripling \
                     --budget 5 \
                     --epoch 20000 \
                     --lr 0.001 \
                     --bs 16 \
                     --n_step 1

#### Test ToupleGDD model

	python main.py --graph test_data/Wiki-2.txt \
                     --model Tripling \
                     --model_file tripling.ckpt \
                     --budget 10 \
                     --test



                    

#### Train S2V-DQN model

	python main.py --graph train_data \
                     --model S2V_DQN \
                     --model_file s2vdqn.ckpt \
                     --budget 5 \
                     --epoch 20000 \
                     --lr 0.001 \
                     --bs 16 \
                     --n_step 1

#### Test S2V-DQN model

	python main.py --graph test_data/Wiki-2.txt \
                     --model S2V_DQN \
                     --model_file s2vdqn.ckpt \
                     --budget 10 \
                     --test

More instructions
-----------------
[Tips](https://github.com/Dtrycode/ToupleGDD/blob/main/instructions.md)

Dependency requirement
----------------------

- Python 3.6.13
- NumPy 1.19.5
- PyTorch 1.10.1+cu102
- PyG (PyTorch Geometric) 2.0.3
- PyTorch Scatter 2.0.9
- Tqdm 4.64.0
- SciPy 1.5.4

Code files
----------

- main.py: load program arguments, graphs and set up RL agent and environment.
- runner.py: conduct simulation, train and test RL agent.
- models.py: define parameters and structures of S2V_DQN and ToupleGDD.  
- rl_agents.py: define agents to follow reinforcement learning procedure.
- environment.py: store the process of simulation.  
- utils/graph_utils.py: utility functions to load graphs, run MonteCarlo/RR to estimate influence spread.   

Reference
---------
Please cite our work if you find our code/paper is useful to your work.

	@article{chen2022touplegdd,
      title={ToupleGDD: A Fine-Designed Solution of Influence Maximization by Deep Reinforcement Learning},
      author={Chen, Tiantian and Yan, Siwen and Guo, Jianxiong and Wu, Weili},
      journal={arXiv preprint arXiv:2210.07500},
      year={2022}
    }


License
-------
This project is licensed under the terms of the [MIT](https://github.com/Dtrycode/ToupleGDD/blob/main/LICENSE) license.
