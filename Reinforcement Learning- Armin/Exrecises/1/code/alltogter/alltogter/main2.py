from env2 import World
import numpy as np
import pandas as pd
# from IPython.display import display


if __name__ == "__main__":

    env = World()

# A. transition model and reward models - dynamic programming

##########################################################################################################################
######################################### question 1 - dynamic programming ###############################################
    '''
    transitions = env.new_transition_models(0.8) # call to our transition model
    rewards = env.new_reward_models(-0.04, 0.8) # call to our reward model

    value, policy = env.value_iterations(transitions,rewards) # call value iterations
    env.plot_value(value) # plot values
    policy = policy.reshape(-1, 1)
    env.plot_policy(policy) # plot policy
    '''
##################################################################################################################
######################################### question 2 - monte carlo ###############################################

    # monte carlo tuning - calculate by MSE rate
    '''
    env.reset()
    # the values by dynamic programming model (values by grid order)
    dp = [0, 0, -0.064, -0.101, 0.439, 0.084, 0.111, -0.064, 0.6, 0.0, 0.208, 0, 0.947, 0.782, 0.611, 0.208, 0, 0.947,0.766, 0]
    MSE_list = np.zeros(9)
    QMSE = np.zeros(9)
    total_episodes = 10000
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09] # optional different alpha values

    indexa = 0
    for alpha in alphas:
        #print("alpha value is:", alpha)
        Q,value, policy= env.mc_control(total_episodes, alpha)
        sum_diff = 0  # sum the differences
        len_policy = len(policy)  # total number of items in list
        for i in range(0, len_policy):
            difference = dp[i] - value[i]  # calculate the difference between dp observed and mc predicted value
            sum_diff = sum_diff + (difference ** 2)  # square the differences and adding to sum of all the differences
        MSE = sum_diff / len_policy  # dividing sum_diff by total values to get average
        #print(MSE)
        MSE_list[indexa] = MSE
        indexa += 1
    # print all MSE by alphas
    for j in range(0, 9):
      print("alphas number", j, "is:", MSE_list[j])


    
    # monte carlo model 
    env.reset()
    total_episodes = 50000
    Q, value, policy = env.mc_control(total_episodes, 0.08) # call monte carlo function with alpha 0.08
    env.plot_value(value) # plot values
    env.plot_qvalue(Q) # plot plot_qvalue
    env.plot_policy(policy) # plot policy
    
'''

############################################################################################################
######################################### question 2 - Sarsa ###############################################

    # Sarsa tuning - calculate by MSE rate
    '''
    env.reset()
    # the values by dynamic programming model (values by grid order)
    dp = [0, 0, -0.064, -0.101, 0.439, 0.084, 0.111, -0.064, 0.6, 0.0, 0.208, 0, 0.947, 0.782, 0.611, 0.208, 0, 0.947, 0.766, 0]
    MSE_list = np.zeros((9))
    QMSE = np.zeros(9)
    total_episodes = 10000
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    indexa = 0
    for alpha in alphas:
        #print("the alpha number is:", alpha)
        Q ,value,policy= env.sarsa(total_episodes, alpha)
        sum_diff_sr = 0  # sum the differences
        len_policy_s = len(policy)  # total number of items in list
        for i in range(0, len_policy_s):
            difference = dp[i] - value[i]  # calculate the difference between dp observed and mc predicted value
            sum_diff_sr = sum_diff_sr + difference ** 2  # square the differences and adding to sum of all the differences
        MSE = sum_diff_sr / len_policy_s  # dividing sum_diff_s by total values to get average
        print(MSE)
        MSE_list[indexa] = MSE
        indexa += 1

    for j in range(0, 9):
      print("alphas number",j,"is:",MSE_list[j])
    '''

    '''
    # Sarsa model
    env.reset()
    total_episodes = 100000
    Q, value, policy = env.sarsa(total_episodes, 0.01) # call Sarsa function with alpha 0.01
    env.plot_value(value) # plot values
    env.plot_qvalue(Q) # plot qvalue
    policy = policy.reshape(-1, 1)
    env.plot_policy(policy) # plot policy

    '''

############################################################################################################
######################################### question 4 - Qlearning ###########################################

    # Qlearning tuning - calculate by MSE rate
    '''
    env.reset()
    # the values by dynamic programming model (values by grid order)
    dp = [0, 0, -0.064, -0.101, 0.439, 0.084, 0.111, -0.064, 0.6, 0.0, 0.208, 0, 0.947, 0.782, 0.611, 0.208, 0, 0.947,0.766, 0]
    MSE_list = np.zeros((9))
    QMSE = np.zeros(9)
    total_episodes = 10000
    alphas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

    indexa = 0
    for alpha in alphas:
        # print("the alphas is:", alpha)
         Q, value,policy=env.Qlearning(total_episodes,alpha)
         sum_diff_q = 0  # sum the differences
         len_policy_q = len(policy)  # total number of items in list
         for i in range(0, len_policy_q):
             difference = dp[i] - value[i] # calculate the difference between dp observed and mc predicted value
             sum_diff_q = sum_diff_q + difference ** 2    # square the differences and adding to sum of all the differences
         MSE = sum_diff_q / len_policy_q # dividing sum_diff_s by total values to get average
         print(MSE)
         QMSE[indexa]=MSE
         indexa+=1

    for j in range(0, 9):
      print("alphas number", j, "is:", QMSE[j])


    # Qlearning model
    env.reset()
    total_episodes = 10
    Q ,value ,policy = env.Qlearning(total_episodes, 0.02) # call Qlearning model
    env.plot_value(value) # plot value
    env.plot_qvalue(Q) # plot qvalue
    policy = policy.reshape(-1, 1)
    env.plot_policy(policy) # plot policy
    
    '''

############################################################################

############### Armin code
    '''
    for n_episode in np.arange(0, num_episodes):
        # reset sets the agent back to the starting state. It can take an argument \exploring starts", for which
        # the agent starts at a random initial state (true) or at the initial state 4 if no argument is provided.
        env.reset()
        done = False
        t = 0
        #show displays the gridworld and the agent in the gridworld
        env.show()
        while not done:
            # render draws the gridworld and the agent in the gridworld
            env.render()
            action = np.random.randint(1, env.nActions ) # take a random action
            next_state, reward, done = env.step(action)  # observe next_state and reward
            print(next_state)
            print(reward)
            print(done)

            #close closes the gridworld
            env.close()
            t += 1
            if done:
                print("Episode",n_episode + 1, "finished after {} timesteps".format(t + 1))
                break

    '''






























