
""" Network """
learning_rate = 1e-6
target_update_interval = 2000

""" Replay buffer """
buffer_size = 10000
initial_buffer_size = 1000
beta_begin = 0.4
beta_end = 1.0
beta_decay = 500000

""" Epsilon """
epsilon_begin = 1.0
epsilon_end = 0.1
epsilon_decay = 10000

""" Other parameters """
num_episodes = 1000000
# num_episodes = 100
gamma = 0.9
batch_size = 32
max_step_episode = 300
print_interval_episode = 10000
# print_interval_episode = 10
save_interval_episode = 100000
# save_interval_episode = 100

""" seed """
seed = 0
