from collections import defaultdict
import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt

def finiteDeck(): 
    cards = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

    deck = np.repeat(cards, 4)
    np.random.shuffle(deck)

    return deck.tolist()

class Agent:
    def __init__(self):
        self.cards = [] # States

    def addCard(self, card):
        self.cards.append(card)


class Game:
    STICK = 0
    HIT = 1

    def __init__(self):
        self.agent = Agent()
        self.returns = defaultdict(list)
        self.valueFunctions = defaultdict(int)
        self.Q = defaultdict(dict)
        self.policies = defaultdict(dict)

        for i in range(1, 22):
            for k in range(0, 2):
                self.Q[i][k] = 0
                self.policies[i][k] = 0.5

    def chooseAction(self, score):
        if score > 21:
            return self.STICK
            
        if score < 11:
            return self.HIT
        
        return np.random.choice(2, p = list(self.policies[score].values()))

    def getCard(self):
        deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        return np.random.choice(deck)

    def sumPoints(self):
        result = sum(self.agent.cards)

        # Ace value.
        if result + 10 <= 21 and 1 in self.agent.cards:
            result += 10

        return result

    def getReward(self, score):
        if score < 17:
            return 0
        if score <= 21:
            return 2
        return -1

    def playEpisode(self):
        self.agent = Agent()

        # Deal cards 
        #for i in range(2):

        self.agent.addCard(self.getCard())

        states = []
        while True:
            state = self.sumPoints()

            if state > 21:
                break

            action = self.chooseAction(state)

            if action == self.HIT:
                self.agent.addCard(self.getCard())

            nextState = self.sumPoints()
            reward = self.getReward(nextState)

            states.append((state, action, reward))

            if action == self.STICK:
                break

        return states

    def monteCarloControl(self, episodes, gamma = 1, epsilon = 1.0):
        for i in range(episodes):
            print(f'{i}')
            states = self.playEpisode()

            epsilon = max(epsilon * 0.99995, 0.0001)

            G = 0
            for k, (state, action, reward) in enumerate(states):
                G = gamma * G + reward

                if not (state, action) in states[0:k]:
                    self.returns[(state, action)].append(G)

                    self.Q[state][action] = np.mean(self.returns[(state, action)])

                    bestAction = max(self.Q[state], key=self.Q[state].get) # Same as argmax but with dictionaries.
                    
                    for action in range(0, 2):
                        if action == bestAction:
                            self.policies[state][action] = 1 - epsilon + (epsilon / 2) # |A(st)| = 2
                        else:
                            self.policies[state][action] = epsilon / 2


startTime = time()

game = Game()
game.monteCarloControl(10000,gamma=1, epsilon=.5)
print(f'Done in: {time() - startTime}')

for policy in game.policies:
    print(f'State {policy}')
    print(game.policies[policy])


# x = [i[0] for i in game.episodes]
# print(x)
# print(len(x))

# y = game.valueFunctions
# print(y)
# print(len(y))

# plt.plot(x, y)
# plt.show()
