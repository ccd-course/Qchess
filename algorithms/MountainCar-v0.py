import gym
from mountaincar_ai import Dqn

env = gym.make('MountainCar-v0')

nb_actions = env.action_space.n
input_size = len(env.observation_space.sample())

model = Dqn(input_size, nb_actions, 0.9)


def start(select_action, print_score, num_episodes=100):
    num_steps = 500
    reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        for t in range(1, num_steps+1):
            action = select_action(reward, state)
            state, reward, done, _ = env.step(int(action))
            if done:
                print(print_score())
                break


start(model.update, model.score)
print(model.score())
