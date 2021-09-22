from onitama.game import BotVsBot, Winner
from onitama.rl import RandomAgent, SimpleAgent

def compate_agents(seed, agent1, agent2, n_eps=100):
    game = BotVsBot(agent1, agent2, seed, verbose=False)
    winner1 = 0
    for i in range(n_eps):
        game.reset()
        winner = Winner.noWin
        while winner == Winner.noWin:
            state = game.stepBot()
            winner = Winner(state["winner"])

        winner1 += winner == Winner.player1
    print("Player 1 won {} / {}".format(winner1, n_eps))


if __name__ == "__main__":
    seed = 1231
    compate_agents(seed, SimpleAgent(seed, isPlayer1=True,threshold=0), RandomAgent(seed, isPlayer1=False))
    compate_agents(seed, SimpleAgent(seed, isPlayer1=True,threshold=1), SimpleAgent(seed, isPlayer1=False))