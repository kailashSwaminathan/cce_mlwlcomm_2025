from typing import Optional
import numpy as np
import gymnasium as gym

class GridWorldEnv(gym.Env):
    
    def __init__(self, size: int = 2):
        self.size = size
        
        self._agent_location = -1
        
        # valid states S0, S1, S2
        self.observation_space = gym.spaces.Discrete(3)
        # valid actions A1, A2
        self.action_space = gym.spaces.Discrete(2)
        
    def _get_obs(self):
        return self._agent_location
        
    def _get_info(self):
        return {}
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        
        self._agent_location = self.np_random.integers(0, self.observation_space.n, dtype=int)
        #self._agent_location = 0 # Always start with S0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action):
        """
        """
        reward = 0
        newloc = -1
        match(self._agent_location):
            case 0: # state S0
                if action == 0:
                    newloc = 1
                    reward = 5
                elif action == 1:
                    newloc = 2
                    reward = 10
            case 1:
                if action in [0,1]:
                    newloc = 1
                    reward = 2
            case 2:
                if action in [0,1]:
                    newloc = 2
                    reward = 0
        
        if newloc != -1:
            self._agent_location = newloc

        observation = self._get_obs()
        info = self._get_info()
        terminated = False
        truncated = False
        
        return observation, reward, terminated, truncated, info
        
gym.register(id="gymnasium_env/qno42", entry_point=GridWorldEnv)

def solution(debug=0):
    env = gym.make('gymnasium_env/qno42')
    
    # Q-Learning parameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    num_episodes = 500
    
    # Initialize Q-table
    qtable = np.zeros((env.observation_space.n, env.action_space.n))
    
    def epsilon_greedy_policy(state, epsilon):
        if np.random.rand() < epsilon:
            if debug: 
                print("Exploring...")
            return env.action_space.sample() # Explore
        else:
            if debug:
                print("Exploiting...")
            return np.argmax(qtable[state])  # Exploit
    
    # Q-Learning algorithm
    for episode in range(num_episodes):
        if debug:
            print(f'Starting episode {episode}')
        state, _ = env.reset()
        done = False
        
        count  = 0
        while count < 10:
            action = epsilon_greedy_policy(state, epsilon)
            if debug:
                print(f"Performing: Action({action}) in State({state})")
            next_state, reward, _, _, _ = env.step(action)
            
            # Update Q-value
            best_next_action = np.argmax(qtable[next_state])
            td_target = reward + gamma * qtable[next_state, best_next_action]
            qtable[state, action] += alpha * (td_target - qtable[state, action])
            state = next_state
            count += 1
        if debug:
            print(qtable)       
    # Display the learned Q-values
    if debug:
        print("Learned Q-values:")
        print(qtable)
    return qtable
    
if __name__ == "__main__":
    solution(debug=1)
            
                