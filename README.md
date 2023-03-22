# **Reinforcement Learning**
<p>Reinforcement Learning is a type of machine learning that enables an agent to learn from its interactions with an environment through trial-and-error learning. The agent learns to take actions in an environment to maximize its reward signal over time. The goal of reinforcement learning is to find the optimal policy that maximizes the cumulative reward received by the agent.</p>

## **Cliff Walking Reinforcement Learning**
<p>The Cliff Walking environment is a classic Reinforcement Learning problem in which an agent must navigate a grid world with a cliff that drops off into the water. The agent must reach the goal on the other side of the cliff while avoiding falling off the cliff. Train a Reinforcement Learning agent to navigate the Cliff Walking environment using Sarsa and Q-Learning algorithms in Python with OpenAI Gym. The goal is to reach the goal state on the other side of the cliff while avoiding falling off the cliff.</p>

## **Sarsa and Q-Learning Algorithms**
<p>Sarsa and Q-Learning are two popular reinforcement learning algorithms used to solve various problems. Both algorithms use the concept of the Q-value function to determine the optimal policy.

Sarsa is an on-policy algorithm, which means it learns the Q-values for the current policy that the agent is following. It uses the Q-value function to estimate the expected reward of taking a particular action in a particular state. Sarsa updates its Q-values after every action and is known to converge to the optimal policy.

Q-Learning is an off-policy algorithm, which means it learns the Q-values for the optimal policy, even if the agent is following a different policy. Q-Learning updates its Q-values after taking an action and observing the resulting state and reward. It is known to converge to the optimal policy, even if the agent explores the environment randomly.</p>

## **Libraries used**
<p>In this project, we have used the following libraries:

- `cv2` : OpenCV is a library of programming functions mainly aimed at real-time computer vision.
- `gym` : Gym is a toolkit for developing and comparing reinforcement learning algorithms.
- `numpy` : NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.
- `pickle` : Python pickle module is used for serializing and de-serializing python object structures.</p>

## **Agent Performance**
<p>I have trained the agent to explore the environment using Sarsa and Q-Learning algorithms. The agent learns to maximize its reward over time by taking actions in the environment.

The following images show the agent exploring the environment and the agent's performance using Sarsa and Q-Learning algorithms :</p>

- <h3>Agent exploring environment</h3>

![Agent Training](https://user-images.githubusercontent.com/96681905/226179019-d7404a91-a046-45f4-9ed0-849ef370f7ce.gif)

- <h3>Agent performance using Sarsa algorithm</h3>

![Sarsa Agent](https://user-images.githubusercontent.com/96681905/226179016-5036e0b5-a664-4124-b607-3c4181da768b.gif)


- <h3>Agent performance using Q-Learning algorithm</h3>

![Q-Learning Agent](https://user-images.githubusercontent.com/96681905/226179020-415413d3-2e69-49b0-a07d-eae33a7f5811.gif)


<p>The agent's performance improves over time as it learns to take actions that maximize its reward signal.</p>


