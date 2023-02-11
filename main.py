import gym
import numpy as np
from ddpg import Agent
import matplotlib.pyplot as plt
import pickle

def plot_learning_curve(x, scores, figure_file):
    ''' Plot the learning curve '''
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def write_to_file(data, file):
    with open(file, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(data, filehandle)

def read_from_file(file):
    with open(file, 'rb') as filehandle:
        # read the data as binary data stream
        data = pickle.load(filehandle)
    return data

def test_files():
    a = [10, 100, 11]
    write_to_file(a, "text.txt")
    print ("the list is: {}".format(read_from_file("text.txt")))

    b = [10, 11, 12]
    write_to_file(b, "text.txt")
    print ("the list is: {}".format(read_from_file("text.txt")))


def model_training(env, agent, n_games, file):
    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        # reset terminal state and score
        done = False
        score = 0
        # initialize random process for action exploration
        observation, _ = env.reset()
        agent.noise.reset()
        while not done:
            # select an action
            action = agent.choose_action(observation)
            # execute the action and get reward, next state
            observation_, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            # store the transition
            agent.remember(observation, action, reward, observation_, term)
            # learn the transition (update networks and target networks)
            agent.learn()
            # aggregate total score
            score += reward\
            # update observation to the current
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
        
        # save score history to file
        write_to_file(score_history, file)

def run_training_episodes():
     # test write and read files
    # test_files()

    n_games = 1000
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    
    # saved files
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'
    log_file = 'plots/' + filename + '.txt'
    
    model_training(env, agent, n_games, log_file)

    score_history = read_from_file(log_file)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)

def run_evaluation_episodes():
     # test write and read files
    # test_files()

    n_games = 1000

    env = gym.make('LunarLanderContinuous-v2', render_mode='human')
    state, _ = env.reset()

    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    
    agent.load_models()
    for i in range(n_games):

        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            state, reward, term, trunc, _ = env.step(action=action)

            score += reward
            done = term or trunc
        
        env.reset()
        print('Done with episode: %d with score: %d' % (i + 1, score))

def main():
    eval = True

    if eval:
        run_evaluation_episodes()
    else:
        run_training_episodes()

if __name__ == '__main__':
    main()