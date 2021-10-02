import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

class GridworldAgent:
    def __init__(self, env, policy, gamma = 0.9, 
                 start_epsilon = 0.9, end_epsilon = 0.1, epsilon_decay = 0.9):
        self.env = env
        self.n_action = len(self.env.action_space)
        self.policy = policy
        self.gamma = gamma
        self.v = dict.fromkeys(self.env.state_space,0)  # state value initiated as 0
        self.n_v = dict.fromkeys(self.env.state_space,0)  # number of actions performed: use it for MC state value prediction
        self.q = defaultdict(lambda: np.zeros(self.n_action))  # action value
        self.n_q = defaultdict(lambda: np.zeros(self.n_action))  # number of actions performed: use it for MC state-action value prediction
        
        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay
    
    def get_epsilon(self,n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay**n_episode),self.end_epsilon)
        return(epsilon)
    
    def get_v(self,start_state,epsilon = 0.):
        episode = self.run_episode(start_state,epsilon)
        """
        Write the code to calculate and return the state value 
        given a deterministic policy. Should return a scalar. Study the components of episode to
        understand how to calculate return.
        YOUR CODE HERE
        """
        _rewards = 0

        for _result in episode:
            _rewards += _result[2]
        
        return _rewards / len(episode)

    
    def get_q(self, start_state, first_action, epsilon=0.):
        episode = self.run_episode(start_state,epsilon,first_action)
        """
        Write the code to calculate and return the action value of a state 
        given a deterministic policy. Should return a scalar. Study the components of episode to
        understand how to calculate return.
        YOUR CODE HERE
        """
        return self.q(start_state, first_action)
    
    def select_action(self,state,epsilon):
        best_action = self.policy[state]
        if random.random() > epsilon:
            action = best_action
        else:
             action = np.random.choice(np.arange(self.n_action))
        return(action)
    
    def print_policy(self):
        for i in range(self.env.sz[0]):
            print('\n----------')
            for j in range(self.env.sz[1]):
                p=self.policy[(i,j)]
                out = self.env.action_text[p]
                print(f'{out} |',end='')
    
    def print_v(self, decimal = 1):
        for i in range(self.env.sz[0]):
            print('\n---------------')
            for j in range(self.env.sz[1]):
                out=np.round(self.v[(i,j)],decimal)
                print(f'{out} |',end='')
    
    def run_episode(self, start, epsilon, first_action = None):
        result = []
        state = self.env.reset(start)
        #dictate first action to iterate q
        if first_action is not None:
            action = first_action
            next_state,reward,done,_ = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: return(result)
        while True:
            action = self.select_action(state,epsilon)
            next_state,reward,done,_ = self.env.step(action)
            result.append((state,action,reward,next_state,done))
            state = next_state
            if done: break
        return(result)
    
    def update_policy_q(self):
        for state in self.env.state_space:
            self.policy[state] = np.argmax(self.q[state])
    
    def mc_predict_v(self,n_episode=10000,first_visit=True):
        for t in range(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start,e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            for i in range(len(transitions)):
                if first_visit and (states[i] not in traversed):
                    """
                    Implement first-visit Monte Carlo for state values(see Sutton and Barto Section 5.1)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """ 
                elif not first_visit:
                     """
                    Implement any-visit Monte Carlo for state values(see Sutton and Barto Section 5.1)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
        for state in self.env.state_space:
            if state != self.env.goal:
                self.v[state] = self.v[state] / self.n_v[state]
            else:
                self.v[state] = 0
    
    def mc_predict_q(self,n_episode=10000,first_visit=True):
        for t in range(n_episode):
            traversed = []
            e = self.get_epsilon(t)
            transitions = self.run_episode(self.env.start,e)
            states,actions,rewards,next_states,dones = zip(*transitions)
            for i in range(len(transitions)):
                if first_visit and ((states[i],actions[i]) not in traversed):
                    """
                    Implement first-visit Monte Carlo for state-action values(see Sutton and Barto Section 5.2)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """
                elif not first_visit:
                    
                    """
                    Implement any-visit Monte Carlo for state-action values(see Sutton and Barto Section 5.2)
                    Comment each line of code with what part of the pseudocode you are implementing in that line
                    YOUR CODE HERE
                    """

        for state in self.env.state_space:
            for action in range(self.n_action):
                if state != self.env.goal:
                    self.q[state][action] = self.q[state][action] / self.n_q[state][action]
                else:
                    self.q[state][action] = 0
        
    def mc_control_q(self,n_episode=10000,first_visit=True):
        """
        Write the code to perform Monte Carlo Control for state-action values
        Hint: You just need to do prediction then update the policy
        YOUR CODE HERE
        """
        raise NotImplementedError
        
    def mc_control_glie(self,n_episode=10000,first_visit=True,lr=0.):
        """
        Bonus: Taking hints from the mc_predict_q and mc_control_q methods, write the code to
        perform GLIE Monte Carlo control. Comment each line of code with what part of the pseudocode you are implementing in that line
        YOUR CODE HERE
        """
        raise NotImplementedError