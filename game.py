from collections import defaultdict
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

    def __init__(self, episodes):
        self.episodes = episodes
        self.previousStates = []
        self.episodeStates = {i:[] for i in range(episodes)}
        self.returns = defaultdict(list)
        self.valueFunctions = defaultdict(int)


    def applyPolicy(self):
        if self.sumPoints() == 20 or self.sumPoints() >= 21:
            return self.STICK 
        return self.HIT

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

    # Generate episodes
    def play(self):
        for episode in range(self.episodes):
            self.agent = Agent()

            # Deal cards 
            #for i in range(2):
            self.agent.addCard(self.getCard())

            while True:
                action = self.applyPolicy()

                reward = self.getReward()

                self.saveEpisodeState(episode, self.sumPoints(), action, reward)
                
                if action == self.STICK:
                    break
                else:
                    self.agent.addCard(self.getCard())
            
            self.monteCarlo(self.episodeStates[episode])

    # Monte carlo First Visit Prediction
    def monteCarlo(self, states):     
        G = 0
        gamma = 1

        for state in states:
            score = state[0] # score

            G = gamma * G + state[2] # reward
            
            self.returns[score].append(G)
            self.valueFunctions[score] = np.mean(self.returns[score])


game = Game(10)
game.play()
print(game.valueFunctions)
# x = [i[0] for i in game.episodes]
# print(x)
# print(len(x))

# y = game.valueFunctions
# print(y)
# print(len(y))

# plt.plot(x, y)
# plt.show()
