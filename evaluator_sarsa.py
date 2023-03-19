import gym
import cv2
import pickle as pkl
import numpy as np
from PIL import Image

cliffEnv = gym.make("CliffWalking-v0")

q_table = pkl.load(open("sarsa_q_table.pkl", "rb"))


# Creates cliff walking grid
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    # Start
    # frame = cv2.putText(img, text="S", org=(49 * 0 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame


# puts the agent at a state
def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action


NUM_EPISODES = 5

frames = []  # for gif

for episode in range(NUM_EPISODES):

    done = False

    total_reward = 0
    episode_length = 0

    frame = initialize_frame()
    state = cliffEnv.reset()

    while not done:
        frame2 = put_agent(frame.copy(), state)

        # Append the frame as a PIL Image
        # frames.append(Image.fromarray(frame2))
        frames.append(Image.fromarray(frame2.astype('uint8')))

        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)
        action = policy(state)
        state, reward, done, _ = cliffEnv.step(action)

        total_reward += reward
        episode_length += 1

    print("Episode:", episode, "Episode Length:",
          episode_length, "Reward:", total_reward)


# Save the frames as a GIF
frames[0].save('Sarsa Agent.gif', format='GIF',
               append_images=frames[1:], save_all=True, duration=250)


cliffEnv.close()
