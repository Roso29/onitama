from onitama.game import PvBot, State, get_move, Piece, PvP, Winner
from onitama.rl import RandomAgent, SimpleAgent
import gym
import numpy as np
import copy


def flip_pos(pos):
    newPos = np.subtract([4, 4], pos)
    return [int(newPos[0]), int(newPos[1])]


def flip_card(card):
    return list(np.flip(np.flip(card, 0), 1))


def flip_player(player):
    player_flipped = copy.deepcopy(player)
    player_flipped.cards = [flip_card(c) for c in player.cards]
    player_flipped.king = Piece(flip_pos(player.king.pos), -1)
    player_flipped.pawns = [Piece(flip_pos(p.pos), i) for i, p in enumerate(player.pawns)]
    return player_flipped


def flip_game_view(game):
    """
    Changes from p1 view (default) to p2 view, as though board has rotated or players swapped seats
    Not same as flipping
    Returns a PvP object with a flipped board (ie. from player 2's view)
    """
    game_flipped = PvP(game.seed, game.verbose, game.playerStart)
    game_flipped.winner = game.winner
    game_flipped.isPlayer1 = game.isPlayer1
    game_flipped.player1 = flip_player(game.player1)
    game_flipped.player2 = flip_player(game.player2)
    game_flipped.spare_card = flip_card(game_flipped.spare_card)
    return game_flipped


def get_board_state(player_dict):
    pawns = np.zeros((5, 5, 1))
    king = np.zeros((5, 5, 1))
    for i, j in player_dict["pawns"]:
        pawns[i][j] = 1
    k, l = player_dict["king"]
    king[k][l] = 1
    return pawns, king


def moveToMask(move, player):
    """
    :return: (a, b, c) tuple the indices of the action for mask = 1
    """
    # 5 x 5 grid is pr(pickup piece here)
    piecePos = player.king.get() if move.isKing else player.pawns[move.i].get()
    # the next is 5 x 5 spaces x 2 cards
    # card id is 0 or 1
    move_ravel = np.ravel_multi_index((piecePos[0], piecePos[1], move.pos[0], move.pos[1], move.cardId),
                                      (5, 5, 5, 5, 2))
    mask = np.unravel_index(move_ravel, (5, 5, 50))
    return mask


def _get_piece(piece_pos, player):
    if np.array_equal(piece_pos, player.king.get()):
        return True, -1
    for i, pawn in enumerate(player.pawns):
        if np.array_equal(piece_pos, pawn.get()):
            return False, i


def get_piece(piece_pos, game, isPlayer1):
    """
    :return: isKing, i
    """
    if isPlayer1:
        return _get_piece(piece_pos, game.player1)
    else:
        return _get_piece(piece_pos, game.player2)


def get_mask(game, isPlayer1, mask_shape=(5, 5, 50)):
    """
    (5 x 5 x 50) (same shape as agent output)
    Returns the mask over valid moves
    Binary tensor.
    """
    game = get_game_maybe_flipped(game, isPlayer1)
    mask = np.zeros(mask_shape)
    player = game.player1 if isPlayer1 else game.player2
    assert len(game.get_valid_moves(player, isPlayer1)) > 0, "No valid moves for masking"
    for move in game.get_valid_moves(player, isPlayer1):
        ac = moveToMask(move, player)
        mask[ac] = 1
        # print("Valid move in mask: {}".format(np.ravel_multi_index(ac, mask_shape)))
    return mask


def _get_obs(game, isPlayer1):
    """
    Returns (5, 5, 9) see above for observation format
    """
    game = get_game_maybe_flipped(game, isPlayer1)
    # see game class for API
    game_state = State(game.get())
    obs = []
    # cards
    obs.append(np.stack(game_state.player1_dict["cards"], -1))
    obs.append(np.stack(game_state.player2_dict["cards"], -1))
    obs.append(np.expand_dims(game_state.spare_card, -1))
    # board
    pawns_p1, king_p1 = get_board_state(game_state.player1_dict)
    obs.append(pawns_p1)
    obs.append(king_p1)
    pawns_p2, king_p2 = get_board_state(game_state.player2_dict)
    obs.append(pawns_p2)
    obs.append(king_p2)
    return np.concatenate(obs, -1)


def actionToMove(ac_chosen, game, isPlayer1, mask_shape):
    """
    :param ac_chosen: reshaped action (piece pos i, piece pos j, board x card data)
    :return: Move() object
    """
    game = get_game_maybe_flipped(game, isPlayer1)
    ac_ravel = np.ravel_multi_index(ac_chosen, mask_shape)
    (piece_pos_i, piece_pos_j, pos_i, pos_j, card_id) = np.unravel_index(ac_ravel, (5, 5, 5, 5, 2))
    # wrap in int() to avoid non-serialisable errors
    piece_pos = [int(piece_pos_i), int(piece_pos_j)]
    pos = [int(pos_i), int(pos_j)]
    if not isPlayer1:
        pos = flip_pos(pos)
    piece = get_piece(piece_pos, game, isPlayer1)
    isKing, i = piece
    move = get_move(pos, isKing, card_id, i)
    return move


def get_game_maybe_flipped(game, isPlayer1):
    return game if isPlayer1 else flip_game_view(game)


def get_reward(game, isPlayer1, reward_dict_id,sparse=False):
    # can get game state by eg.
    # game.player1

    player = game.player1 if isPlayer1 else game.player2
    opponent = game.player2 if isPlayer1 else game.player1

    next_moves = game.get_valid_moves(player, isPlayer1, show_overlapping_moves=True)
    opp_moves = game.get_valid_moves(opponent, not isPlayer1, show_overlapping_moves=True)

    # We have a winner
    reward_win = 0
    if game.winner is not Winner.noWin:
        # 1 for p1, 2 for p2
        curPValue = (1 + int(not isPlayer1))
        if game.winner is Winner.draw:
            reward_win = 0
        elif game.winner.value == curPValue:
            reward_win = 1
        else:
            reward_win = -1

    if sparse: return reward_win

    # Get number of rows moved
    move_forwards = 0
    move_pawn_forwards = 0
    if player.last_move is not None:
        rows_moved = player.last_move.pos[0] - player.last_pos[0]
        row_orientation = 1 if not isPlayer1 else -1
        move_forwards = max(0, rows_moved * row_orientation)
        if not player.last_move.isKing:
            move_pawn_forwards = move_forwards

    # Threatening a win by king move
    shrine_win_possible = 0
    enemy_home = [0, 2] if isPlayer1 else [4, 2]
    if player.last_move is not None:
        for move in next_moves:
            if move.isKing and move.pos == enemy_home:
                shrine_win_possible = 1

    # King is attackable
    unsafe_king = 0
    if player.king.pos in [move.pos for move in opp_moves]:
        unsafe_king = 1


    # There is an undefended attackable pawn
    undefended_attackable_pawn_penalty = 0
    for pawn in player.pawns:
        if pawn.pos in [move.pos for move in opp_moves] and pawn.pos not in [move.pos for move in next_moves]:
            undefended_attackable_pawn_penalty = 1

    # Pawn went to a defendable square (without sacrificing defense of another pawn)
    defendable_squares = [move.pos for move in next_moves]
    defended_pawn_move = 0
    if player.last_move is not None:
        if player.last_move.pos in defendable_squares and not player.last_move.isKing:
            defended_pawn_move = 1

    # Threatened pawn capture
    threatened_pawn_captures = 0
    if player.last_move is not None:
        for pawn in opponent.pawns:
            if pawn.pos in [move.pos for move in next_moves]:
                threatened_pawn_captures = 1

    # Threatened safe pawn captures
    safe_threatened_pawn_captures = 0
    for pawn in opponent.pawns:
        if pawn.pos in [move.pos for move in next_moves] and pawn.pos not in [move.pos for move in opp_moves]:
            safe_threatened_pawn_captures += 1

    # Move threatens a king capture on the next move
    threatened_king_capture = 0
    if opponent.king.pos in [move.pos for move in next_moves]:
        threatened_king_capture = 1

    # Whether the move made resulted in a capture
    pawn_taken = 0
    if opponent.lost_pawn_last_move:
        pawn_taken = 1

    # Whether a pawn was lost
    pawn_lost = 0
    if player.lost_pawn_last_move:
        pawn_lost = 1

    reward_weights=[
    {   # old
        "move_forwards": 0.01,
        "move_pawn_forwards": 0,
        "defended_pawn_move": 0,
        "unsafe_king": 0,
        "undefended_attackable_pawn_penalty": 0,
        "threatened_pawn_captures": 0,
        "safe_threatened_pawn_captures": 0,
        "threatened_king_capture": 0,
        "pawn_taken": 0.1,
        "pawn_lost": -0.1,
        "shrine_win_possible": 0,
        "win": 1,
        "game_duration_penalty": 0
    },

    {   # version 1
        "move_forwards": 0,
        "move_pawn_forwards": 0.01,
        "defended_pawn_move": 0,
        "unsafe_king": -0.5,
        "undefended_attackable_pawn_penalty": -0.05,
        "threatened_pawn_captures": 0,
        "safe_threatened_pawn_captures": 0.05,
        "threatened_king_capture": 0.05,
        "pawn_taken": 0.25,
        "pawn_lost": -0.25,
        "shrine_win_possible": 0.5,
        "win": 1,
        "game_duration_penalty": -0.01
    },

    {   # version 2
        "move_forwards": 0,
        "move_pawn_forwards": 0.01,
        "defended_pawn_move": 0.025,
        "unsafe_king": -0.5,
        "undefended_attackable_pawn_penalty": -0.05,
        "threatened_pawn_captures": 0.025,
        "safe_threatened_pawn_captures": 0.05,
        "threatened_king_capture": 0.05,
        "pawn_taken": 0.25,
        "pawn_lost": -0.25,
        "shrine_win_possible": 0.5,
        "win": 1,
        "game_duration_penalty": -0.01
    }]

    reward_dict = {
        "move_forwards": move_forwards,
        "move_pawn_forwards": move_pawn_forwards,
        "unsafe_king": unsafe_king,
        "undefended_attackable_pawn_penalty": undefended_attackable_pawn_penalty,
        "defended_pawn_move": defended_pawn_move,
        "threatened_pawn_captures": threatened_pawn_captures,
        "safe_threatened_pawn_captures": safe_threatened_pawn_captures,
        "threatened_king_capture": threatened_king_capture,
        "pawn_taken": pawn_taken,
        "pawn_lost": pawn_lost,
        "shrine_win_possible": shrine_win_possible,
        "win": reward_win,
        "game_duration_penalty": 1
    }

    reward_weights = reward_weights[reward_dict_id]

    reward = 0
    for k, r in reward_dict.items():
        reward += r * reward_weights[k]
    return reward


class OnitamaEnv(gym.Env):
    """
    Defaults to player 1
    See README for obs and ac space definitions
    """

    def __init__(self, seed, agent_type=SimpleAgent, reward_dict=0, isPlayer1=True, verbose=True):
        super(OnitamaEnv, self).__init__()
        self.game = PvBot(agent_type(seed), seed, verbose=verbose)
        self.observation_space = gym.spaces.Box(np.zeros((5, 5, 59)), np.ones((5, 5, 59)))
        self.action_space = gym.spaces.Discrete(5 * 5 * 25 * 2)
        self.mask_shape = (5, 5, 50)
        self.isPlayer1 = isPlayer1
        self._seed = seed
        self.reward_dict = reward_dict

    def step(self, ac):
        ac = np.squeeze(ac)
        # action is index into 5 x 5 x 50
        ac = np.unravel_index(ac, self.mask_shape)
        move = actionToMove(ac, self.game, self.isPlayer1, self.mask_shape)

        self.game.step(move)
        self.game.stepBot()

        info = {}
        done = False
        # win, lose or draw
        if self.game.winner is not Winner.noWin:
            done = True
            info["winner"] = self.game.winner.value
            # success if controlled winning player
            info["is_success"] = self.game.winner.value == (1 + int(not self.isPlayer1))
        return self.get_obs(), get_reward(self.game, self.isPlayer1, reward_dict_id=self.reward_dict), done, info

    def reset(self):
        self.game.reset()
        # if it's not the rl's turn then let the bot take it's turn
        if self.game.isPlayer1 != self.isPlayer1:
            self.game.stepBot()
        return self.get_obs()

    def get_obs(self):
        """
        Observation and mask for valid actions
        :return:
        """
        return np.concatenate([_get_obs(self.game, self.isPlayer1), get_mask(self.game, self.isPlayer1)], -1)

    def seed(self, seed):
        self._seed = seed
        np.random.seed(seed)


class OnitamaSelfPlayEnv(gym.Env):
    """
    An env where step and reward is called for each player 1 and 2 being RL
    Assume p1 to be the main player for training
    """

    def __init__(self, seed, verbose=True, deterministicSelfPlay=False, reward_dict=0):
        super(OnitamaSelfPlayEnv, self).__init__()
        self.game = PvP(seed, verbose=verbose)
        self.observation_space = gym.spaces.Box(np.zeros((5, 5, 59)), np.ones((5, 5, 59)))
        self.action_space = gym.spaces.Discrete(5 * 5 * 25 * 2)
        self.mask_shape = (5, 5, 50)
        self._seed = seed
        # the model weights for self play
        self.selfPlayModel = None
        self.deterministicSelfPlay = deterministicSelfPlay
        self.isPlayer1 = True
        self.reward_dict=reward_dict

    def step(self, ac):
        # run the training model action
        move = self.getMove(ac)
        self.game.step(move)
        # step the self play action
        acSelfPlay, _ = self.selfPlayModel.predict([self.get_obs()], deterministic=self.deterministicSelfPlay)
        moveSelfPlay = self.getMove(acSelfPlay)
        self.game.step(moveSelfPlay)

        info = {}
        done = False
        # win, lose or draw
        if self.game.winner is not Winner.noWin:
            done = True
            info["winner"] = self.game.winner.value
            # success if controlled winning player
            info["is_success"] = self.game.winner.value == (1 + int(not self.isPlayer1))
        return self.get_obs(), get_reward(self.game, self.isPlayer1, reward_dict_id=self.reward_dict,sparse=True), done, info

    def getMove(self, ac):
        ac = np.squeeze(ac)
        # action is index into 5 x 5 x 50
        ac = np.unravel_index(ac, self.mask_shape)
        move = actionToMove(ac, self.game, self.game.isPlayer1, self.mask_shape)
        return move

    def reset(self):
        assert self.selfPlayModel, "No model set"
        self.game.reset()
        # if it's not the rl's turn then let the self play opponent take it's turn
        if self.game.isPlayer1 != self.isPlayer1:
            # step the self play action
            acSelfPlay, _ = self.selfPlayModel.predict([self.get_obs()], deterministic=self.deterministicSelfPlay)
            moveSelfPlay = self.getMove(acSelfPlay)
            self.game.step(moveSelfPlay)
        return self.get_obs()

    def render(self, mode='human'):
        pass

    def get_obs(self):
        """
        Observation and mask for valid actions
        :return:
        """
        return np.concatenate([_get_obs(self.game, self.game.isPlayer1), get_mask(self.game, self.game.isPlayer1)], -1)

    def seed(self, seed):
        self._seed = seed
        np.random.seed(seed)

    def setSelfPlayModel(self, model):
        self.selfPlayModel = model
