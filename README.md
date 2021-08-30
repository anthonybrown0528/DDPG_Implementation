## Overview
This is an implementation in Pytorch of [the paper](https://arxiv.org/abs/1509.02971) titled "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING". The algorithm is called DDPG (Deep Deterministic Policy Gradient). Features in this RL algorithm are:

* Deep reinforcement learning for continuos action spaces
* Actor-Critic Architechture, model-free algorithm and deterministic policy
* Find policy which competitive performance comparing with those found by planning algorithms (with have a full acess to the environment dynamics)

## Set up requirement
- Install Pytorch, gym, numpy and matplotlib
- Run `main.py`

## Backgroud of the algorithm
Off-policy learning is a process where behavior policy is different than the policy used during learning process. As in deterministic policy, the expected value of action-value only depends on the environment (the reward), it is possible to apply off-policy learning where the training is conducted with transitions sampled from different behavior policy.

Q learning is a common off-policy algorithm which uses greedy policy (choose action with has maximum value). However, this approach was perceived to unstable if using large, non-linear function approximators. Recent advances allowed a scale-up Q Learning which use large neural network as a function approximators for action values functions in Deep Q Network (DQN).

On the other hand, it is not straight forward to apply Q-Learning for continuous action spaces. A workaround on this matter is using an actor-critic approach based on the DPG (Deterministic Policy Gradient) algorithm. In DPG, the actor function is parameterised and specifies the current policy by mapping states to specific actions.

## Implementation

The algothrim leveraged the sucess of Deep Q Network (DQN) which has been shown a human level performance on many Atari games and extended it in order to solve problems required continuos action spaces without discretizing them. It is a combination of deterministic policy gradient (DPG) and DQN such as:

* Use DQN innovatives for learning value functions such as (1) The network is trained off-policy using replay buffer memory and with a target Q Network (2) Batch normalisation
* Actor-Critic architechture of DPG

The algorithm steps:
The Actor maintained a parameterised actor function which is the current policy (Mapping states to a specific action). The Actor is updated by finding a policy gradient using chain rule of expected return with respect to the actor parameters. Following are steps of the the calculation:

- Randomly sample a state from memory
- Use Actor network to determine action
- Plug action to Critic network to get the action-value
- Take gradient of this value according to Actor parameters

The replay buffer is a finite sized cache contained transitions during interaction with the environment. These are used to update the Actor and Critic in each time-step. The learned networks are not directly copied to the target networks but update them in a "soft update" rule.

Batch nomalisation technique is used to manually scale the features. This is to ensure they have a similar range across environments and units. This practice is applied for the state input and all layers of the networks prior
to the action input.

In off-policy algorithms, we can treat the problem of exploration independently from the learning algorithm. From the actor policy, noise sample from noise processes can be added to the actor policy.

# Psedo-code for the algorithm
The main steps of training algorithm in each episode are:
* Initialize a random process
* Get initial start state
* Until reach the terminal state do:
    * Select an action using the current policy (Actor) and noise
    * Get reward and new state from this action
    * Store this transition to the replay buffer
    * Sample a minibatch from replay buffer for learning
    * Get expected action-values of the minibatch (these values are derived from target networks)
    * Update the Critic networks with the new evaluations by miniizing loss functions
    * Update Actor policy using sampled policy gradient
    * Update the target networks from the trained Actor and Critic network