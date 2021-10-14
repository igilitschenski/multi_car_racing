"""
This script is used to evaluate a trained agent and save test scores.
TODO for your agent:
    1. Implement your tester (see example in A3C/a3c.py)
    2. Import your tester agent below
    3. Fill in the name of your tester agent in get_agent_for_algo below
    4. To test your model run
        python benchmarking.py --algo [choose from 'a3c', 'dqn', 'ddpg', 'dddqn', 'ppo']
    5. Results of your run will be saved to ./results
Comments:
    - Currently number of testing episodes is set to 3 so that everybody can try it. We will increase this to ~1000
    - You can use the flag --render to show your agent while testing
    - You can set number of testing episodes via --num_test_episodes
"""
from model_tester import *
from argparse import ArgumentParser
import os, sys
import numpy as np
sys.path.insert(0, '../algorithms')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from A3C.a3c import A3CTesterAgent
from dueling.DDQN import DDQNTesterAgent
# Add your TesterAgent import here


def get_agent_for_algo(algo):
    if algo == 'a3c':
        return A3CTesterAgent
    elif algo == 'ppo':
        pass
    elif algo == 'dqn':
        pass
    elif algo == 'ddpg':
        pass
    elif algo == 'ddqn':
        return DDQNTesterAgent

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--algo",
        choices=("a3c", "dqn", "ddpg", "ddqn", "ppo"),
        type=str,
        help="Select algorithm.",
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        help="Number of episodes to run.",
        default=5
    )
    parser.add_argument(
        "--render",
        action='store_true',
        default=False,
        help="Render episodes during test.",
    )
    args = parser.parse_args()

    agent = get_agent_for_algo(args.algo)()
    tester = ModelTester(agent=agent,
                         num_test_episodes=args.num_test_episodes,
                         render=args.render,
                         )
    print('Testing {} algorithm...'.format(args.algo))
    eval_data = tester.evaluate()
    log_file = 'results/{}.log'.format(args.algo)
    np_file = 'results/{}.npy'.format(args.algo)
    print('Testing finished, saving results to {} and {}...'.format(log_file, np_file))
    with open(log_file, 'w') as f:
        f.write('number of test episodes: {}\n'.format(eval_data['num_episodes']))
        f.write('average score: {}\n'.format(eval_data['avg_score']))
        f.write('std of score: {}\n'.format(eval_data['std_score']))
    np.save(np_file, eval_data)


