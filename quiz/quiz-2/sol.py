import sys
import numpy as np
import random
import qlearn

def sol_42():
    """
    """
    # Parameters
    lr = 0.1
    gamma = 0.9
    e_greedy = 0.1
    numeps = 500
    maxsteps = 10
    actions = ['A0','A1']
    def get_next(s, a):
        """
        """
        next_s = ''
        match (s):
            case 'S0':
                if a == 'A0':
                    next_s = 'S1'
                elif a == 'A1':
                    next_s = 'S2'
            case 'S1' | 'S2':
                if a in ['A0','A1']:
                    next_s = s
        return next_s
    
    rewards = {
        "S0" : { "A0" : 5, "A1" : 10 },
        "S1" : { "A0" : 2, "A1" : 2 },
        "S2" : { "A0" : 0, "A1" : 0 }
    }
    qtable = {
        "S0" : { "A0" : 0, "A1" : 0 },
        "S1" : { "A0" : 0, "A1" : 0 },
        "S2" : { "A0" : 0, "A1" : 0 }
    }
    for episode in range(numeps):
        print(f"Starting episode {episode}")
        s = 'S0'
        for i in range(maxsteps):
            rnum = np.random.rand()
            if rnum < e_greedy:
                print("Exploring...")
                a = random.sample(actions,1)[0]
            else:
                print("Exploiting...")
                keys, vals = [], []
                for k,v in qtable[s].items():
                    keys.append(k)
                    vals.append(v)
                indx = np.argmax(vals)
                a = keys[indx]
            print(f"Performing: Action({a}) in State({s})")
            prev = qtable[s][a]
            next_s = get_next(s, a)
            newv = prev + lr * (rewards[s][a] + gamma * max([qtable[next_s][a] for a in ['A0','A1']]) - prev)            
            qtable[s][a] = newv
            s = next_s
            print(qtable)
    print(qtable)

def main(qno):
    """
    """
    match(qno):
        case '42':
            qtable = qlearn.solution()
            indx = np.where(qtable == np.max(qtable))
            print(f"{qno}. State  : {'S0' if indx[0][0] == 0 else 'S1' if indx[0][0] == 1 else 'S2' if indx[0][0] == 2 else 'ERROR'}" + \
                  f"    Action : {'A0' if indx[1][0] == 0 else 'A1' if indx[1][0] == 1 else 'ERROR'}")
        case _:
            print("NOT IMPLEMENTED!!")
            
if __name__ == "__main__":
    main(sys.argv[1])