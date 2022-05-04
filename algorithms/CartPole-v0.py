import gym
from cartpole_ai import Dqn

env = gym.make('CartPole-v1')

nb_actions = env.action_space.n
input_size = len(env.observation_space.sample())

model = Dqn(input_size, nb_actions, 0.9)


def start(select_action, num_episodes=100):
    num_steps = 500
    reward = 0
    for episode in range(num_episodes):
        state = env.reset()
        env.render()
        for t in range(1, num_steps+1):
            action = select_action(reward, state)
            state, reward, done, _ = env.step(int(action))
            if done:
                break


start(model.update)
print(model.score())
