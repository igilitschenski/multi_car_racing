import matplotlib.pyplot as plt
import numpy as np
import os
print(os.environ['PATH'])
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

algos = ['hc', 'a3c', 'ddpg', 'ddqn', 'dqn', 'ppo']
bar_labels = ['hard-coded', 'A3C', 'DDPG', 'DDQN', 'DQN', 'PPO']
colors = ['orangered', 'forestgreen', 'royalblue', 'darkorange', 'gold', 'mediumpurple']
bar_x = []
bar_y = []
bar_err = []
fig = plt.figure(0)
for i, algo in enumerate(algos):
    data = np.load('../results/' + algo + '.npy', allow_pickle=True)[()]
    scores = data['scores']
    num_episodes = data['num_episodes']
    assert num_episodes == 1000
    mean = np.mean(scores)
    ste = np.std(scores) / np.sqrt(num_episodes)
    bar_x.append(i)
    bar_y.append(mean)
    bar_err.append(ste)
plt.bar(x=np.stack(bar_x),
        height=np.stack(bar_y),
        width=0.7,
        color=colors,
        yerr=np.stack(bar_err),
        capsize=4.0
        )
ax = plt.gca()
ax.set_xticks(np.stack(bar_x))
ax.set_xticklabels(bar_labels)

plt.xlabel('Algorithm', fontsize=20)
plt.ylabel('Score', fontsize=20)
fig.set_dpi(200)
plt.savefig('benchmark_bar_stde.png')
plt.show()