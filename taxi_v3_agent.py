import numpy as np
from collections import defaultdict
import random

class TaxiAgent:
    def __init__(self, env, gamma = 0.8, alpha = 1e-1,
                 start_epsilon = 1, end_epsilon = 1e-2, epsilon_decay = 0.999):
        
        self.env = env
        self.n_action = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        
        #action values
        self.q = defaultdict(lambda: np.zeros(self.n_action)) #action value
        
        #epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    #get epsilon
    def get_epsilon(self,n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return(epsilon)
    
    #select action based on epsilon greedy
    def select_action(self,state,epsilon):
        #implicit policy; if we have action values for that state, choose the largest one, else random
        best_action = np.argmax(self.q[state]) if state in self.q else self.env.action_space.sample()
        if random.random() > epsilon:
            action = best_action
        else:
             action = self.env.action_space.sample()
        return(action)
    
    def on_policy_td_sarsa(self, state, action, reward, next_state, n_episode):
        """
        Implement On policy TD learning or SARSA
        YOUR CODE HERE
        """
        _old_q_vals = self.getQValue(state, action)
        _reward = self.alpha * reward

        # update q vals w.r.t. old q vals and reward
        self.q[(state, action)] = (1 - self.alpha) * _old_q_vals + _reward

        # check if next state exists for the given state and action
        # if next state exists then add it's val too to q_val
        if next_state:
            _action = self.select_action(n_episode)
            self.q[(
                state, action
            )] += self.alpha * self.discount * self.getQValue(next_state, _action)
        
    def off_policy_td_q_learning(self, state, action, reward, next_state):
        """
        Implement Off policy TD learning ie SARSA-MAX/Q learning 
        YOUR CODE HERE
        """
        raise NotImplementedError
    
    def getQValue(self, state, action):
        return self.q[state, action]
        