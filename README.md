# simplerl
simplerl is a package of deep reinforcement learning (RL) algorithms using PyTorch and modeled for compatibility with OpenAI Gym environments. Unlike many other RL packages, simplerl is designed so that users can easily customize agents and their underlying models. The focus here is simplicity and modularity. All agents share a similar framework with two main methods - action and update - for compatibility with a single training algorithm, which can be heavily customized using hooks. The agent design is based on mixins as many popular RL components can be used across multiple agents, such as using prioritized experience replay for DQN, DDPG, TD3, etc. 

To illustrate simplerl usage and benchmark the different agents, each agent undergoes one or more tests on OpenAI Gym's Lunar Lander environments. This is a challenging enough task where traditional RL algorithms struggle, but simple enough that an agent can be trained on a CPU in a reasonable amount of time. Check out the iPython Notebooks for additional details.

## Development Roadmap
Since this package is still in its infancy, there are many items on development roadmap, including:
1. Distributional DQNs (C51 and QR-DQN)
2. Distributional critics for actor-critic algorithms
3. A2C (simplerl uses multiprocessing for distributed training, but does not currently support asynchronous architectures so no A3C)
4. PPO
5. Recurrent network support (maybe!)
6. More documentation!

## Requirements
| Name            | Version   | Notes                       |
|-----------------|-----------|-----------------------------|
| cloudpickle     | 1.2.2     |                             |
| gym             | 0.15.3    | For the Lunar Lander tests  |
| numpy           | 1.16.5    |                             |
| pytorch         | 1.3.0     |                             |
