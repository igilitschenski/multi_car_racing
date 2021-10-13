# Source: https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
import os, sys
sys.path.append(os.path.dirname(sys.path[0]))
project_dir = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(project_dir)
sys.path.append(os.path.join(project_dir, 'gym_multi_car_racing'))
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import gym_multi_car_racing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow.python.compiler.mlcompute import mlcompute



#Hyperparameters
learning_rate = 0.0001 #0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 1#5#3
T_horizon     = 5#100 #100 #20

num_frames = 5
beta = 0.001
SCALE = 10.0  # Track scale
PLAYFIELD = 2000 / SCALE  # Game over boundary
max_temp = 5.0
ACTIONS = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Action Space Structure
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),  # (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),  # Range        -1~1       0~1   0~1
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)
        ]
VELOCITY_REWARD_PROPORTIONAL = 0.0
VELOCITY_REWARD_LOW = -10.0
ANGLE_DIFF_REWARD = 0.0
ON_GRASS_REWARD = -1.0
BACKWARD_REWARD = 0.0

n_epi_train   = 800
n_epi_test = 10

def to_grayscale(img):
    return np.dot(img, [0.299, 0.587, 0.144])


def zero_center(img):
    return (img - 127.0) / 256.0


def crop(img, bottom=12, left=6, right=6):
    height, width = img.shape
    return img[0: height - bottom, left: width - right]


def preprocess_state(s, car_id):
    s = s[car_id, ...]
    s = to_grayscale(s)
    s = zero_center(s)
    s = crop(s)
    tensor_im = tf.convert_to_tensor(s.astype(np.float32))
    return tf.expand_dims(tf.expand_dims(tensor_im, 0),0)  # N, C, H, W tensor


def add_frame(s_new, s_stack):
    s_stack = tf.roll(s_stack, shift=-1, axis=1)
    s_stack_np = s_stack.numpy()
    s_stack_np[:, -1, :, :]  = s_new.numpy()
    s_stack = tf.convert_to_tensor(s_stack_np, dtype=tf.float32)
    return s_stack

def action_space_setup(env):
    env.cont_action_space = ACTIONS
    env.action_space = gym.spaces.Discrete(len(env.cont_action_space))

def get_reward(env, action):
    step_reward = np.zeros(env.num_agents)
    for car_id, car in enumerate(env.cars):  # First step without action, called from reset()

        velocity = abs(env.all_feats[car_id, 47])
        step_reward[car_id] += (VELOCITY_REWARD_LOW if velocity < 2.0 else velocity * VELOCITY_REWARD_PROPORTIONAL)  # normalize the velocity later
        step_reward[car_id] += abs(env.all_feats[car_id, 3]) * ANGLE_DIFF_REWARD  # normalize angle diff later
        step_reward[car_id] += float(env.all_feats[car_id, 4]) * ON_GRASS_REWARD  # driving on grass
        step_reward[car_id] += float(env.all_feats[car_id, 5]) * BACKWARD_REWARD  # driving backward
    return step_reward

def episode_end(env, car_id):
    done = False
    x, y = env.cars[car_id].hull.position
    if (abs(x) > PLAYFIELD or abs(y) > PLAYFIELD) or (env.driving_backward[car_id]):
        done = True
    return done


class PPO(tf.keras.Model):
    def __init__(self, num_of_inputs):
        super(PPO, self).__init__()
        #self.data = []
        #input_shape = (1, 5, 84, 84)
        self.conv1 = Conv2D(filters=16, kernel_size=(5,5), strides=(4,4), padding='valid', use_bias=False, activation='relu')
        #self.conv2 = Conv2D(filters=16, kernel_size=(3,3), strides=(4,4), padding='valid', use_bias=False, activation='relu') #this is being problematic... 
        self.linear1 = Dense(256, input_shape=(320,), activation=None)
        self.policy = Dense(12, input_shape=(256,), activation=None)
        self.value = Dense(1, input_shape=(256,), activation=None)
        self.num_of_actions = 12
        self.optimizer = keras.optimizers.Adam(lr=learning_rate)

    #samples actions from policy given states
    def pi(self, x, t=1.0): #x is inputted images, t is temperature control
        conv1_out = self.conv1(x)
        #conv2_out = self.conv2(conv1_out) #this is being problematic... 
        #flattened = Flatten()(conv2_out)
        flattened = Flatten()(conv1_out)
        linear1_out = self.linear1(flattened)
        policy_output = self.policy(linear1_out)
        prob = tf.nn.softmax(policy_output / t, axis=-1)
        if tf.reduce_any(tf.math.is_nan(prob)):
                    print("prob has a nan")
        return prob
    
    # value estimates
    def v(self, x): # x is inputted images
        conv1_out = self.conv1(x)
        #conv2_out = self.conv2(conv1_out) #this is being problematic... 
        flattened = Flatten()(conv1_out)  
        linear1_out = self.linear1(flattened)
        v = tf.squeeze(self.value(linear1_out), -1)
        return v

def train(self):

    env = gym.make("MultiCarRacing-v0", 
                    num_agents = 1, 
                    verbose = 1,
                    direction='CCW',
                    use_random_direction = True, 
                    backwards_flag = True, 
                    h_ratio = 0.25,
                    use_ego_color = False, 
                    setup_action_space_func = action_space_setup,
                    get_reward_func = get_reward,
                   episode_end_func=episode_end,
                   observation_type='frames')

    model_path = "./saved_models/PPO/state_dict_model.pt"
    #model.load_state_dict(torch.load(model_path))
    
    scores = []
   
    print_interval = 5
    save_interval = 50

    for n_epi in range(n_epi_train):
        done = False
        score = 0.0
        s = env.reset()

        # Initialize state with repeated first frame
        car_id = 0
        s = preprocess_state(s, car_id)
        s_prime = s
        s = tf.repeat(s, 5, axis=1)
        sprime_frames = s
        temperature = np.exp(-(n_epi - n_epi_train) / n_epi_train * np.log(max_temp))    
        action_list = []

        while not done:
            s_lst, a_lst, r_lst, sprime_lst, prob_lst, done_lst = [], [], [],[], [], []
            for t in range(T_horizon):
                s = add_frame(s_prime, s)
                prob = model.pi(s, t=temperature)

                m = tfp.distributions.Categorical(probs=prob)
                a = m.sample().numpy().item()
                #m = tf.random.categorical(prob, num_samples=1)
                action_vec = np.array(ACTIONS[a])
                s_prime, r, done, _ = env.step(action_vec)
                r = r[0]
                r /= 100
                s_prime = preprocess_state(s_prime, car_id)

                # Store the states, sprimes,  actions, rewards, probs, 
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r.astype(np.float32))
                prob_lst.append(prob[0,a].numpy().item())
                done_mask = 0 if done else 1
                done_lst.append([done_mask])
                env.render()
                action_list.append(a)
                sprime_frames = add_frame(s_prime, sprime_frames)
                sprime_lst.append(sprime_frames)
                score += r
                if done:
                    break

            
            s_batch, sprime_batch, a_batch = tf.concat(s_lst, axis=0), tf.concat(sprime_lst, axis=0), tf.squeeze(tf.convert_to_tensor(a_lst), -1)
            sprime_values = self.v(sprime_batch)



            with tf.GradientTape() as tape:
                td_target = tf.convert_to_tensor(r_lst) + gamma * sprime_values* tf.squeeze(tf.convert_to_tensor(done_lst, dtype=tf.float32), -1)
                s_values = self.v(s_batch)
                delta = td_target - s_values
                delta = delta.numpy()

                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = gamma * lmbda * advantage + delta_t
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = tf.squeeze(tf.convert_to_tensor(advantage_lst, dtype=tf.float32))

                pi = self.pi(s_batch, t=temperature)
                if tf.reduce_any(tf.math.is_nan(pi)):
                    print("pi has a nan")
                pi_a = tf.gather_nd(pi, tf.expand_dims(a_batch,-1), 1)
                ratio = tf.math.exp(tf.math.log(pi_a) - tf.math.log(tf.convert_to_tensor(prob_lst)))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1-eps_clip, 1+eps_clip) * advantage

                #Build loss function
                entropy_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pi * tf.math.log(pi + 1e-20), axis=1))
                policy_loss = tf.math.reduce_mean(tf.math.minimum(surr1, surr2))
                value_loss = tf.keras.losses.Huber() 
                loss = -1.0 * policy_loss + \
                    0.5 * value_loss(self.v(s_batch).numpy(), td_target.numpy()) + \
                    -beta* entropy_loss


            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            

            # if n_epi%save_interval == 0:
            #     torch.save(model.state_dict(), model_path)


        scores.append(score)
        #print(action_list)

        #if n_epi%save_interval == 0:
            #torch.save(model.state_dict(), model_path)

        if n_epi%print_interval == 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score))
            print(action_list)
            print(prob)
    
    env.close()

    #torch.save(model.state_dict(), model_path)
    return scores, action_list

# TODO: convert test function to tf
def test():
    
    env = gym.make("MultiCarRacing-v0", num_agents=1, direction='CCW',
                   use_random_direction=True, backwards_flag=True, h_ratio=0.25,
                   use_ego_color=False)
    model = PPO()
    # Load saved model
    model_path = "./saved_models/PPO/state_dict_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    score = 0.0
    print_interval = 1
    scores = []

    for n_epi in range(n_epi_test):
        score = 0.0
        done = False
        env.reset()
        s = env.get_feat(car_id=0)
        action_list = []
        while not done:
            
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                action = env.cont_action_space[a]
                action_list.append(a)
                _, r, done, _ = env.step(action)
                s_prime = env.get_feat(car_id=0)
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime
                score += r.item()
                env.render()

        scores.append(score)
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, score))
        print(action_list)
        #if n_epi%print_interval==0 and n_epi!=0:
            #print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            #score = 0.0
        
    env.close()
    return scores, action_list

if __name__ == '__main__':
    model = PPO(num_frames)
    scores, action_list = train(model)
    #print(scores)
    print(action_list)
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.show()