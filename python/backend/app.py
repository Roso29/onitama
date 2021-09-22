from flask import Flask, request
from onitama.game import PvP, PvBot, BotVsBot
from onitama.rl import RandomAgent, SimpleAgent, RLAgent, DQNMaskedCNNPolicy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

seed = 12442

twoPlayer = PvP(seed)
againstBot = PvBot(SimpleAgent(seed), seed)
againstRL = PvBot(RLAgent(seed, "selfplay.zip", algorithm="PPO", isPlayer1=False), seed)

#botVsBot = BotVsBot(RandomAgent(seed, isPlayer1=True),
 #                   RLAgent(seed, "../onitama-py/onitama/rl/logs/dqn-tb/2021_03_24-12_06_17_PM/best_model.zip", algorithm="DQN", isPlayer1=False),
  #                  seed)
# botVsBot = BotVsBot(RandomAgent(seed, isPlayer1=True),
#                     SimpleAgent(seed, isPlayer1=False),
#                     seed)
# botVsBot = BotVsBot(SimpleAgent(seed, isPlayer1=True, isVerbose=True),
#                     SimpleAgent(seed, isPlayer1=False, isVerbose=True),
#                     seed)
game = againstRL

games = [twoPlayer, againstBot]#, botVsBot]
game_id = 2


@app.route('/getState')
def get_current_state():
    return game.get()


@app.route('/sendMove', methods=['GET', 'POST'])
def handle_move():
    data = request.json
    return game.stepApi(data)


@app.route('/stepBot')
def step_bot():
    return game.stepBot()


@app.route('/reset')
def reset():
    game.reset()
    return game.get()


@app.route('/toggleGameMode')
def toggle_game_mode():
    global game, games, game_id
    # swtich
    game_id = (game_id + 1) % len(games)
    game = games[game_id]
    return game.get()
