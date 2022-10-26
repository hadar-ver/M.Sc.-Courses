import numpy as np
import matplotlib.pyplot as plt
import _env 
import pandas as pd
import copy
from collections import defaultdict

class World(_env .Hidden):

    def __init__(self):

        self.nRows = 4
        self.nCols = 5
        self.stateInitial = [4]
        self.stateTerminals = [1, 2,  10, 12, 17, 20]
        self.stateObstacles = [1, 2,  10, 12, 20]
        self.stateHoles = [1, 2,  10, 12, 20]
        self.stateGoal = [17]
        self.nStates = 20
        self.nActions = 4

        self.observation = [4]  # initial state


    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def _plot_world(self):

        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        stateGoal      = self.stateGoal
        coord = [[0, 0], [nCols, 0], [nCols, nRows], [0, nRows], [0, 0]]
        xs, ys = zip(*coord)
        plt.plot(xs, ys, "black")
        for i in stateObstacles:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            plt.fill(xs, ys, "0.3")
            plt.plot(xs, ys, "black")
        for i in stateTerminals:
            #print("stateTerminal", i)
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.6")
            plt.plot(xs, ys, "black")
        for i in stateGoal:
            (I, J) = np.unravel_index(i-1, shape=(nRows, nCols), order='F')
            I = I + 1
            J = J
            #print("I,J = ", I,J)
            coord = [[J, nRows - I],
                     [J + 1, nRows - I],
                     [J + 1, nRows - I + 1],
                     [J, nRows - I + 1],
                     [J, nRows - I]]
            xs, ys = zip(*coord)
            #print("coord", xs,ys)
            plt.fill(xs, ys, "0.9")
            plt.plot(xs, ys, "black")
        plt.plot(xs, ys, "black")
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        plt.plot(X, Y, 'k-')
        plt.plot(X.transpose(), Y.transpose(), 'k-')

    @staticmethod
    def _truncate(n, decimals=0):
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier

    def plot(self):
        """
        plot function
        :return: None
        """
        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        self._plot_world()
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.5, j - 0.5, str(states[k]), fontsize=26, horizontalalignment='center', verticalalignment='center')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_value(self, valueFunction):

        """
        plot state value function V

        :param policy: vector of values of size nStates x 1
        :return: None
        """

        nStates = self.nStates
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateObstacles:
                    plt.text(i + 0.5, j - 0.5, str(self._truncate(np.round(valueFunction[k],4),3)), fontsize=16, horizontalalignment='center', verticalalignment='center')
                k += 1
        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right',verticalalignment='bottom')
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_policy(self, policy):

        """
        plot (stochastic) policy

        :param policy: matrix of policy of size nStates x nActions
        :return: None
        """
        # remove values below 1e-6
        policy = policy * (np.abs(policy) > 1e-6)

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateObstacles = self.stateObstacles
        stateTerminals = self.stateTerminals
        # policy = policy.reshape(nRows, nCols, order="F").reshape(-1, 1)
        # generate mesh for grid world
        X, Y = np.meshgrid(range(nCols + 1), range(nRows + 1))
        # generate locations for policy vectors
        # print("X = ", X)
        X1 = X.transpose()
        X1 = X1[:-1, :-1]
        # print("X1 = ", X1)
        Y1 = Y.transpose()
        Y1 = Y1[:-1, :-1]
        # print("Y1 =", Y1)
        X2 = X1.reshape(-1, 1) + 0.5
        # print("X2 = ", X2)
        Y2 = np.flip(Y1.reshape(-1, 1)) + 0.5
        # print("Y2 = ", Y2)
        # reshape to matrix
        X2 = np.kron(np.ones((1, nActions)), X2)
        # print("X2 after kron = ", X2)
        Y2 = np.kron(np.ones((1, nActions)), Y2)
        # print("X2 = ",X2)
        # print("Y2 = ",Y2)
        # define an auxiliary matrix out of [1,2,3,4]
        mat = np.cumsum(np.ones((nStates, nActions)), axis=1).astype("int64")
        # print("mat = ", mat)
        # if policy vector (policy deterministic) turn it into a matrix (stochastic policy)
        # print("policy.shape[1] =", policy.shape[1])
        if policy.shape[1] == 1:
            policy = (np.kron(np.ones((1, nActions)), policy) == mat)
            policy = policy.astype("int64")
        #    print("policy inside", policy)
        # no policy entries for obstacle and terminal states
        index_no_policy = stateObstacles + stateTerminals
        index_policy = [item - 1 for item in range(1, nStates + 1) if item not in index_no_policy]
        # print("index_policy", index_policy)
        # print("index_policy[0]", index_policy[0:2])
        mask = (policy > 0) * mat
        # print("mask", mask)
        # mask = mask.reshape(nRows, nCols, nCols)
        # X3 = X2.reshape(nRows, nCols, nActions)
        # Y3 = Y2.reshape(nRows, nCols, nActions)
        # print("X3 = ", X3)
        # print arrows for policy
        # [N, E, S, W] = [up, right, down, left] = [pi, pi/2, 0, -pi/2]
        alpha = np.pi - np.pi / 2.0 * mask
        # print("alpha", alpha)
        # print("mask ", mask)
        # print("mask test ", np.where(mask[0, :] > 0)[0])
        self._plot_world()
        for i in index_policy:
            # print("ii = ", ii)
            ax = plt.gca()
            # j = int(ii / nRows)
            # i = (ii + 1 - j * nRows) % nCols - 1
            # index = np.where(mask[i, j] > 0)[0]
            index = np.where(mask[i, :] > 0)[0]
            # print("index = ", index)
            # print("X2,Y2", X2[ii, index], Y2[ii, index])
            h = ax.quiver(X2[i, index], Y2[i, index], np.cos(alpha[i, index]), np.sin(alpha[i, index]), color='b')
            # h = ax.quiver(X3[i, j, index], Y3[i, j, index], np.cos(alpha[i, j, index]), np.sin(alpha[i, j, index]),0.3)

        # label states by numbers
        states = range(1, nStates + 1)
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                plt.text(i + 0.92, j - 0.92, str(states[k]), fontsize=11, horizontalalignment='right',
                         verticalalignment='bottom')
                k += 1
        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def plot_qvalue(self, Q):
        """
        plot Q-values

        :param Q: matrix of Q-values of size nStates x nActions
        :return: None
        """
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal
        stateObstacles = self.stateObstacles

        fig = plt.plot(1)

        self._plot_world()
        k = 0
        for i in range(nCols):
            for j in range(nRows, 0, -1):
                if k + 1 not in stateHoles + stateObstacles + stateGoal:
                    #print("Q = ", Q)
                    plt.text(i + 0.5, j - 0.15, str(self._truncate(Q[k, 0], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='top', multialignment='center')
                    plt.text(i + 0.9, j - 0.5, str(self._truncate(Q[k, 1], 3)), fontsize=8,
                             horizontalalignment='right', verticalalignment='center', multialignment='right')
                    plt.text(i + 0.5, j - 0.85, str(self._truncate(Q[k, 2], 3)), fontsize=8,
                             horizontalalignment='center', verticalalignment='bottom', multialignment='center')
                    plt.text(i + 0.1, j - 0.5, str(self._truncate(Q[k, 3], 3)), fontsize=8,
                             horizontalalignment='left', verticalalignment='center', multialignment='left')
                    # plot cross
                    plt.plot([i, i + 1], [j - 1, j], 'black', lw=0.5)
                    plt.plot([i + 1, i], [j - 1, j], 'black', lw=0.5)
                k += 1

        plt.title('gridworld', size=16)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def get_nrows(self):

        return self.nRows

    def get_ncols(self):

        return self.nCols

    def get_stateTerminals(self):

        return self.stateTerminals

    def get_stateHoles(self):

        return self.stateHoles

    def get_stateObstacles(self):

        return self.stateObstacles

    def get_stateGoal(self):

        return self.stateGoal

    def get_nstates(self):

        return self.nStates

    def get_nactions(self):

        return self.nActions

    def step(self,action):

        nStates = self.nStates # 4
        stateGoal = self.get_stateGoal() # [17]
        stateTerminals = self.get_stateTerminals() # [1, 2,  10, 12, 17, 20]

        state = self.observation[0] # observation - initial state = grid 4
        # generate reward and transition model
        p_success = 0.8
        r = -0.04
        self.get_transition_model(p_success)
        self.get_reward_model(r,p_success)
        Pr = self.transition_model
        R = self.reward

        prob = np.array(Pr[state-1, :, action])
       # print("prob =", prob)
        next_state = np.random.choice(np.arange(1, nStates + 1), p = prob)
       # print("state = ", state)
       # print("next_state inside = ", next_state)
       # print("action = ", action)
        reward = R[state-1, next_state-1, action]
        #print("reward = ", R[:, :, 0])
        observation = next_state

        #if (next_state in stateTerminals) or (self.nsteps >= self.max_episode_steps):
        if (next_state in stateTerminals):
            done = True
        else:
            done = False
        self.observation = [next_state]
        return observation, reward, done

    def reset(self, *args):
        nStates = self.nStates
        if not args:
            observation = self.stateInitial
        else:
            observation = []
            while not (observation):
                observation = np.setdiff1d(np.random.choice(np.arange(1, nStates +  1, dtype = int)), self.stateHoles + self.stateObstacles + self.stateGoal)
        self.observation = observation

    def render(self):

        nStates = self.nStates
        nActions = self.nActions
        nRows = self.nRows
        nCols = self.nCols
        stateHoles = self.stateHoles
        stateGoal = self.stateGoal

        observation = self.observation #observation
        state = observation[0]


        J, I = np.unravel_index(state - 1, (nRows, nCols), order='F')



        J = (nRows -1) - J



        circle = plt.Circle((I+0.5,J+0.5), 0.28, color='black')
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_artist(circle)

        self.plot()

    def close(self):
        plt.pause(0.3) #0.5
        plt.close()

    def show(self):
        plt.ion()
        plt.show()

##########################################################################################################################
######################################### question 1 - dynamic programming ###############################################

# new_reward_models - calculate the rewards by p of 0.8 and reward of every move step as -0.04
    def new_reward_models(self, r=-0.04, p=0.8):

        reward = [-1, -1, r, r, r, r, r, r, r, -1, r, -1, r, r, r, r, 1, r, r, -1] # mapping the rewards by the grid world index
        nstates = self.get_nstates() # 20
        nstop = self.get_stateTerminals() # [1, 2,  10, 12, 17, 20]
        actions = ["N", "S", "E", "W"]
        # mark the grid world corners
        Nc = [1, 5, 9, 13, 17]
        Wc = [1, 2, 3, 4]
        Ec = [17, 18, 19, 20]
        Sc = [4, 8, 12, 16, 20]
        rewardsa = {}
        for action in actions: # calculate for every action ["N", "S", "E", "W"]
            rewards = np.zeros(nstates) # array of 20 zeros
            for i in range(1, nstates + 1): # 1 to 21 iterations
                if (i in nstop):
                    rewards[i - 1] = 0
                else:
                    if (action == "N"):
                        if (i not in Nc):
                            rewards[i - 1] = p * reward[i - 2]
                        else:
                            rewards[i - 1] = p * reward[i - 1]
                        if (i not in Ec):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i + 3]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
                        if (i not in Wc):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 5]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
                    if (action == "S"):
                        if (i not in Sc):
                            rewards[i - 1] = p * reward[i]
                        else:
                            rewards[i - 1] = p * reward[i - 1]
                        if (i not in Ec):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i + 3]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
                        if (i not in Wc):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 5]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
                    if (action == "E"):
                        if (i not in Ec):
                            rewards[i - 1] = p * reward[i + 3]
                        else:
                            rewards[i - 1] = p * reward[i - 1]
                        if (i not in Nc):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 2]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
                        if (i not in Sc):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
                    if (action == "W"):
                        if (i not in Wc):
                            rewards[i - 1] = p * reward[i - 5]
                        else:
                            rewards[i - 1] = p * reward[i - 1]
                        if (i not in Nc):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 2]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
                        if (i not in Sc):
                            rewards[i - 1] += ((1 - p) / 2) * reward[i]
                        else:
                            rewards[i - 1] += ((1 - p) / 2) * reward[i - 1]
            rewardsa[action] = pd.DataFrame(rewards)
        return rewardsa

# new_transition_models - calculate the transition model by p of 0.8 for N and 0.1 for W or E
    def new_transition_models(self, p=0.8):

        actions = ["N", "E", "S", "W"]
        nstates = self.get_nstates() # 20
        nrows = self.get_nrows() # 4
        nstop = self.get_stateTerminals() # [1, 2,  10, 12, 17, 20]
        # mark the available states corners by the grid world
        Nc = [5, 9, 13]
        Wc = [3, 4]
        Ec = [18, 19]
        Sc = [4, 8, 16]
        transition_models = {} # create transition models
        for action in actions: # calculate for every action ["N", "S", "E", "W"]
            transition_model = np.zeros((nstates, nstates))
            for i in range(1, nstates + 1):
                if i not in nstop:
                    if action == "N":
                        if i not in Nc:
                            transition_model[i - 1][i - 2] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if i not in Wc:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if i not in Ec:
                            transition_model[i - 1][i - 1 + nrows] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                    if action == "S":
                        if i not in Sc:
                            transition_model[i - 1][i] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if i not in Wc:
                            transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if i not in Ec:
                            transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                    if action == "E":
                        if i not in Ec:
                            transition_model[i - 1][i + nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if i not in Nc:
                            transition_model[i - 1][i - 2] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if i not in Sc:
                            transition_model[i - 1][i] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                    if action == "W":
                        if i not in Wc:
                            transition_model[i - 1][i - nrows - 1] += p
                        else:
                            transition_model[i - 1][i - 1] += p
                        if i not in Nc:
                            transition_model[i - 1][i - 2] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                        if i not in Sc:
                            transition_model[i - 1][i] += (1 - p) / 2
                        else:
                            transition_model[i - 1][i - 1] += (1 - p) / 2
                else:
                    transition_model[i - 1][i - 1] = 1
            transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1), columns=range(1, nstates + 1))
        return transition_models

# value_iterations algorithm implementation   - gamma=0.9 -  theta=0.00001
    def value_iterations(self, transition_models, rewards, gamma=0.9, theta=0.00001):

        nstates = self.get_nstates()
        nstop = self.get_stateTerminals()
        value = np.zeros(nstates)
        policy = np.zeros(nstates)
        actions = ["N", "S", "E", "W"]
        delta = 1
        while delta > theta: # stop condition
            delta = 0
            current_value = copy.deepcopy(value) # copy the values of the current state
            for i in range(1, nstates + 1):
                value[i - 1], policy[i - 1] = self.get_the_max(transition_models, rewards, gamma, i, current_value, actions, nstop)
                delta = max(delta, np.abs(current_value[i - 1] - value[i - 1])) # check the improvement rate (delta)
        return value, policy

# get the max value
    def get_the_max(slef, transition_models, rewards, gamma, i, current_value, actions, nstop):

        max_per = {key: 0 for key in actions}
        max_a = ""
        actions = {"N": 1, "E": 2, "S": 3, "W": 4}
        for action in actions:
            max_per[action] = 0
            if i not in nstop:
                max_per[action] += rewards[action].loc[i - 1, 0] + gamma * np.dot(transition_models[action].loc[i, :].values, current_value)
        max_val = -10000
        for k in max_per:
            if max_per[k] > max_val:
                max_a = k
                max_val = max_per[k]
        return max_val, actions[max_a]

############################################################################################################
######################################### question 3 - Sarsa ###############################################

    # Function to choose the next action
    def next_action(self, Q, epsilon, state):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(range(0, self.nActions))
        else:
            action = np.argmax(Q[state - 1, :])
        return action

    # sarsa model
    def sarsa(self,total_episodes,alpha,gamma=0.9):
        Q=np.zeros((self.nStates, self.nActions))
        epsilon = 1

        for episode in range(total_episodes):
            print(episode)
            np.random.seed(episode) # get random episode
            state1 = self.reset()

            if state1 is None:
                state1= self.stateInitial[0]
            action1 = self.next_action(Q,epsilon,state1)

            while True:
                state2, reward, done = self.step(action1) # call step function
                action2 = self.next_action(Q,epsilon,state2)  # take the next action
                predict = Q[state1 - 1, action1]  # Learning the Q-value
                target = reward + gamma * Q[state2 - 1, action2]
                Q[state1 - 1, action1] = predict + alpha * (target - predict)
                state1 = state2
                action1 = action2
                if done:
                    break

          # epsilon = 1/(episode/(total_episodes/10)+1) # epsilon option 1
            epsilon = 1 / ((episode / 10) + 1) # epsilon option 2
          # epsilon = 1(1+sqrt(episode)/10) # epsilon option 3

        policy = np.zeros(self.get_nstates())
        value = np.zeros(self.get_nstates())

        # prepare the value, Q, policy for plot_value, plot_qvalue, plot_policy functions
        for i in range(self.get_nstates()):
            policy[i] = (np.argmax(Q[i, :]) + 1)
            value[i] = (np.max(Q[i, :]))
        return Q, value, policy

############################################################################################################
######################################### question 4 - Qlearning ###########################################

    def Qlearning(self,total_episodes,alpha,gamma=0.9):
        Q = np.zeros((self.nStates, self.nActions))
        epsilon = 1

        for episode in range(total_episodes):
            print(episode)
            np.random.seed(episode) # get random episode
            state1 = self.reset()
            if state1 is None:
                state1 = 4
            action1 = self.next_action(Q, epsilon, state1)
            while True:
                state2, reward, done = self.step(action1) # call step function
                action2 = self.next_action(Q, epsilon, state2) # take the next action
                predict = Q[state1 - 1, action1] # Learning the Q-value
                target = reward + gamma * np.max(Q[state2 - 1, :])
                Q[state1 - 1, action1] = predict + alpha * (target - predict)
                state1 = state2
                action1 = self.next_action(Q, epsilon, state1)
                if done:
                    break

            # epsilon = 1/(episode/(total_episodes/10)+1) # epsilon option 1
            epsilon = 1 / ((episode / 10) + 1)  # epsilon option 2
            # epsilon = 1(1+sqrt(episode)/10) # epsilon option 3

        policy = np.zeros(self.get_nstates())
        value = np.zeros(self.get_nstates())
        for i in range(self.get_nstates()):
            policy[i] = (np.argmax(Q[i, :]) + 1)
            value[i] = (np.max(Q[i, :]))

        return Q, value, policy

##################################################################################################################
######################################### question 2 - monte carlo ###############################################


    def mc_control(self, total_episodes, alpha, gamma=0.9):

        n_actions = self.nActions
        Q = defaultdict (lambda: np.zeros (n_actions))

        for i in range (1, total_episodes + 1):
           # epsilon = 1 / ((episode / 10) + 1) # epsilon option 1
           # epsilon = 1(1+sqrt(i)/10) # epsilon option 2
            epsilon =  1/(i / (total_episodes / 10) + 1) # epsilon option 3

            episode = self.new_episodes(Q, epsilon, n_actions) # call to new_episodes function
            Q = self.new_Q(episode, Q, alpha, gamma) # call to update_Q function
            print(i)

        #prepare the value_state, state_action_Q, optimal_policy for plot_value, plot_qvalue, plot_policy functions
        Q[4] = Q[None]
        del Q[None]
        for st in self.stateTerminals:
            Q[st] = [0, 0, 0, 0]
        sorted_q = sorted (Q.items ())
        value = []
        Q = []
        policy = []
        for k, y in sorted_q:
            value.append (max (y))
        for k, y in sorted_q:
            Q.append (y)
        Q = np.array (Q)
        policy_direction = [np.argmax (x) for x in Q]
        for i, state in enumerate (Q):
            state = [0, 0, 0, 0]
            state[policy_direction[i]] = 1
            policy.append (state)

        return Q, value, policy

# epsilon griddy policy by action
    def policy_per_state(self, Q_s, epsilon):
        n_actions = self.nActions
        policy_per_state = np.ones (n_actions) * epsilon / n_actions # a probability
        a_star = np.argmax (Q_s) #  get the a* action with high val (greedy)
        policy_per_state[a_star] = 1 - epsilon + (epsilon / n_actions) # update a* probability to be: 1-epsilon (+esp/ number of actions)
        return policy_per_state

    def new_episodes(self, Q, epsilon, n_actions): ## generate new episode according to optimal policy
        episode = []
        state = self.reset ()
        if state == None:
            state = 4
        while True:
            if state in Q: # if sate is in q
                action = np.random.choice(np.arange (n_actions), p=self.policy_per_state (Q[state], epsilon)) # take action by probability
            else:
                action=np.random.randint (0, n_actions) # if state is no in Q - take random action
            next_state, reward, done = self.step (action) # step function
            episode.append ((state, action, reward)) # update the state, action and reward to the episode
            state = next_state
            if done:
                break
        return episode

# update q
    def new_Q(self, episode, Q, alpha, gamma):
        states, actions, rewards = zip (*episode) # splite by states actions rewards
        episode_length=len (states)+1
        discounts = np.array ([gamma ** i for i in range (episode_length)])
        for i, state in enumerate (states):
            if state not in states[:i]:
                Q[state][actions[i]] =Q[state][actions[i]] + alpha * (sum (rewards[i:] * discounts[:-(1 + i)]) - Q[state][actions[i]])
        return Q