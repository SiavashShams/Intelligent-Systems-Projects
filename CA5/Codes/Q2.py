import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import product

# parameters
capacity = 20
new_costumer_reward = 10
gamma = 0.9
extra_cap_penalty = -2
lambda_buy_A = 3
lambda_end_A = 3
lambda_buy_B = 4
lambda_end_B = 2
# random customers starting or ending contracts
A_end = [math.pow(lambda_end_A, i) * math.pow(np.e, -lambda_end_A) / math.factorial(i) for i in range(capacity + 1)]
A_buy = [math.pow(lambda_buy_A, i) * math.pow(np.e, -lambda_buy_A) / math.factorial(i) for i in range(capacity + 1)]
B_end = [math.pow(lambda_end_B, i) * math.pow(np.e, -lambda_end_B) / math.factorial(i) for i in range(capacity + 1)]
B_buy = [math.pow(lambda_buy_B, i) * math.pow(np.e, -lambda_buy_B) / math.factorial(i) for i in range(capacity + 1)]
A_prob = np.zeros(41)
B_prob = np.zeros(41)
# probabilities of each event
for j in range(capacity + 1):
    for i in range(capacity + 1):
        event_prob_A = A_end[j] * A_buy[i]
        A_prob[i - j + capacity] = A_prob[i - j + capacity] + event_prob_A
        event_prob_B = B_end[j] * B_buy[i]
        B_prob[i - j + capacity] = B_prob[i - j + capacity] + event_prob_B
prob_table = np.multiply(np.array([A_prob]).T, np.array([B_prob]))
prob_table = np.rot90(prob_table, 2)


# solves Bellman's equation
def calculate_value(V, state_A, state_B, action):
    A_next_state = state_A - action
    B_next_state = state_B + action
    current_V = [[] for _ in range(41)]
    action_V = float(abs(action)) * extra_cap_penalty
    action_reward = [[] for _ in range(41)]
    for i in range(A_next_state - capacity, A_next_state + capacity + 1):
        for j in range(B_next_state - capacity, B_next_state + capacity + 1):
            temp_i = (lambda x: 0 if (x < 0) else 20 if x > 20 else x)(i)
            temp_j = (lambda x: 0 if (x < 0) else 20 if x > 20 else x)(j)
            new_costumers_A = max(0, A_next_state - temp_i)
            new_costumers_B = max(0, B_next_state - temp_j)
            reward = new_costumer_reward * (new_costumers_A + new_costumers_B)
            current_V[i - (A_next_state - capacity)].append(V[temp_i][temp_j])
            action_reward[i - (A_next_state - capacity)].append(reward)
            # calculate V(s,a) with Bellman equation
        action_V += np.dot(prob_table[i - (A_next_state - capacity)],
                           gamma * (np.array(current_V[i - (A_next_state - capacity)]) + np.array(
                               action_reward[i - (A_next_state - capacity)])))
    return action_V


def policy_iteration():
    policy = np.zeros([capacity + 1, capacity + 1]).tolist()
    V = np.zeros([capacity + 1, capacity + 1]).tolist()
    max_iter = 50
    # iterate each policy until convergence or maximum iterations reached
    for i in range(max_iter):
        next_V = np.zeros([capacity + 1, capacity + 1])
        next_policy = np.zeros(np.shape(policy)).tolist()
        # evaluate V(s,a) for each policy
        for i, j in product(range(capacity + 1), range(capacity + 1)):
            states_V = []
            states_action = []
            for k in range(max([-5, -j, -(capacity - i)]), min([i, capacity - j, 5]) + 1):
                action_V = calculate_value(V, i, j, k)
                states_V.append(action_V)
                states_action.append(k)
            # improve policy
            max_state = max(states_V)
            p = states_action[states_V.index(max_state)]
            next_policy[i][j] = p
            next_V[i][j] += max_state
        policy_changed = not (next_policy == policy)
        policy = next_policy
        V = next_V
        # check for convergence
        if not (policy_changed):
            break

    return next_policy


if __name__ == '__main__':
    p = policy_iteration()
    print(np.array(p))
    plt.matshow(p)
    plt.show()
