from random import randint
import numpy as np
import matplotlib.pyplot as plt

n = 69
coincidence = 0
for i in range(10000):
    birthdays = [randint(1, 365) for _ in range(n)]
    if len(birthdays) != len(np.unique(birthdays)):
        coincidence += 1

print("Probability that at least two people have same birthday in a group of 69 is: ", coincidence / 10000)
event_prob = []
for n in range(1, 100):
    coincidence = 0
    for i in range(10000):
        birthdays = [randint(1, 365) for _ in range(n)]
        if len(birthdays) != len(np.unique(birthdays)):
            coincidence += 1
    event_prob.append(coincidence / 10000)

plt.plot(event_prob)
plt.xlabel("N")
plt.ylabel("Probability of Event")
plt.show()
