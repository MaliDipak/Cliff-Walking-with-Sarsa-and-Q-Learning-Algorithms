# Sarsa algorithm
import gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make("CliffWalking-v0")

q_table = np.zeros(shape=(48, 4))


def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action


# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9

NUM_EPISODE = 500

for episode in range(NUM_EPISODE):

    done = False

    total_reward = 0
    episode_length = 0

    state = cliffEnv.reset()
    action = policy(state, EPSILON)

    while not done:

        next_state, reward, done, _ = cliffEnv.step(action)
        next_action = policy(next_state, EPSILON)

        q_table[state][action] += ALPHA * \
            (reward + GAMMA * q_table[next_state]
             [next_action]-q_table[state][action])

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1

    print("Episode:", episode, "Episode Length: ",
          episode_length, "Total Reward: ", total_reward)

cliffEnv.close()


pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training Complete...Q Table Saved :)")
