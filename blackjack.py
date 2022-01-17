from collections import defaultdict
import sys
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
        self.episodes = 0
        self.episodeStates = defaultdict(list)
        self.returns = defaultdict(list)
        self.valueFunctions = defaultdict(int)
        self.Q = defaultdict(dict)
        self.policies = defaultdict(dict)

        for i in range(0, 22):
            for k in range(0, 2):
                self.Q[i][k] = 0
                self.policies[i][k] = 0.5

        # self.policies = {i: 0.5 for i in range(11, 22) }

    def chooseAction(self, state, policy):
        if self.sumPoints() < 11:
            return self.STICK

        return np.random.choice(2, p = list(policy))

    def getCard(self):
        cards = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])
        return np.random.choice(cards)

    def sumPoints(self):
        result = sum(self.agent.cards)

        # Ace value.
        if result > 21 and 11 in self.agent.cards:
            result -= 10

        return result

    def getReward(self):
        if self.sumPoints() <= 21:
            return 1
        return 0

    def saveEpisodeState(self, episode, score, action, reward):
        self.episodeStates[episode].append((score, action, reward)); 

    def playEpisode(self, episode):
        self.agent = Agent()

        # Deal cards 
        #for i in range(2):
        print(f'Playing episode {episode}')

        while True:
            self.agent.addCard(self.getCard())
            score = self.sumPoints()
            
            action = self.chooseAction(score, self.policies[score])
            reward = self.getReward()

            self.saveEpisodeState(episode, self.sumPoints(), action, reward)
            
            if action == self.STICK:
                break

        return self.episodeStates[episode]

    def monteCarloControl(self, episodes, epsilon = 1.4):
        gamma = 1
        epsilon = max(epsilon * 0.9999, 0.01)

        for i in range(episodes):
            states = self.playEpisode(i)

            G = 0
            for state in states:
                score = state[0] # score
                action = state[1] # action

                G = gamma * G + state[2] # reward

                self.returns[(score, action)].append(G)

                self.Q[score][action] = np.mean(self.returns[(score, action)])

                bestAction = max(self.Q[score], key=self.Q[score].get)
                
                print(self.policies[score][action])

                for action in range(0, 2):
                    if action == bestAction:
                        self.policies[score][action] = 1 - epsilon + (epsilon / 2) # |A(st)| = 2
                    else:
                        self.policies[score][action] = epsilon / 2

    # Monte carlo First Visit Prediction
    def firstVisitMonteCarlo(self, states):     
        G = 0
        gamma = 1

        for state in states:
            score = state[0] # score

            G = gamma * G + state[2] # reward

            self.returns[score].append(G)
            self.valueFunctions[score] = np.mean(self.returns[score])

game = Game()
game.monteCarloControl(1000)

# x = [i[0] for i in game.episodes]
# print(x)
# print(len(x))

# y = game.valueFunctions
# print(y)
# print(len(y))

# plt.plot(x, y)
# plt.show()
