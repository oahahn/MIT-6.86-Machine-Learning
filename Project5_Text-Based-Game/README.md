## Project Objectives

The aim of this project was to learn control policies for text-based games using reinforcement learning. In these games, all interactions between players and the virtual world were through text. The current world state is described by elaborate text, and the underlying state is not directly observable. Players read descriptions of the state and respond with natural language commands to take actions. 

In order to design an autonomous game player, a reinforcement learning framework was employed to learn command policies using game rewards as feedback.

Part of the code for this project was written by the course staff. Most of my work can be seen in:

1. **agent_tabular.py** where I implemented the tabular Q-learning algorithm for a simple setting where each text description is associated with a unique index
2. **agent_linear.py** where I implemented the Q-learning algorithm with linear approximation architecture, using bag-of-words representation for textual state description
3. **agent_dqn.py** where I implemented a deep Q-network using PyTorch
