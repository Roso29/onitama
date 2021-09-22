import numpy as np
from enum import Enum
from onitama.game.cards import get_init_cards, card_stamps, seed_cards

KING_ID = -1


# Pawn ids = [0, 4]

class Piece:
    def __init__(self, pos, id):
        self.id = id
        self.pos = pos

    def get(self):
        return self.pos

    def move(self, pos):
        self.pos = pos


class Move:
    """
    Parses json to move object
    """
    def __init__(self, json):
        self.pos = json["pos"]  # [row, col]
        self.isKing = json["name"] == "king"  # T/F
        self.i = -1 if self.isKing else int(json["i"])  # [0-4] for pawn, KING_ID for king
        self.cardId = int(json["id"])  # 0 / 1 for card

    def __str__(self):
        return "(Move : pos {}, isKing {}, i {}, cardId {})".format(self.pos, self.isKing, self.i, self.cardId)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.pos == other.pos and self.isKing == other.isKing \
               and self.i == other.i and self.cardId == other.cardId


def get_move(pos, isKing, cardId, i):
    return Move({
        "pos": pos,
        "name": "king" if isKing else "pawn",
        "i": i,
        "id": cardId
    })


class State:
    """
    Parses JSON sent to FE to object for easier handling
    """

    def __init__(self, json):
        self.player1_dict = json["player1"]
        self.player2_dict = json["player2"]
        self.current_player = json["player"]
        self.spare_card = json["spare_card"]
        self.winner = json["winner"]
        self.mode = json["mode"]

    def __str__(self):
        return "State : current player {}, winner: {}, mode: {}\n" \
               "spare_card: {}\nplayer 1: {}\nplayer2: {}".format(self.current_player, Winner(self.winner).name,
                                                                  self.mode, self.spare_card, self.player1_dict,
                                                                  self.player2_dict)

class Winner(Enum):
    noWin = 0
    player1 = 1
    player2 = 2
    draw = 3

class Player:
    def __init__(self, isPlayer1, cards):
        if isPlayer1:
            row = 4
            self.player = "player1"
        else:
            row = 0
            self.player = "player2"
        self.cards = cards
        # init pieces
        self.king = Piece([row, 2], KING_ID)
        self.pawns = [Piece([row, i + 1 if i >= 2 else i], i) for i in range(4)]
        self.lost_pawn_last_move = False
        self.last_move = None
        
    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {self.player: {"king": self.king.get(), "pawns": [p.get() for p in self.pawns], "cards": self.cards}}

    def step(self, move, card):
        """
        Updates piece and card objects, validation etc. done in Game class
        """
        # Store the last position of the most recently moved piece
        if move.isKing:
            self.last_pos = self.king.get()
            self.king.move(move.pos)
        else:  # it's pawn
            self.last_pos = self.pawns[move.i].get()
            self.pawns[move.i].move(move.pos)
        # Store the most recent move
        self.last_move = move
        # swap card
        self.cards[int(move.cardId)] = card


class PvP:
    def __init__(self, seed, verbose=True, startingPlayer=1):
        self.verbose = verbose
        self.seed = seed
        seed_cards(seed)
        self.mode = "P vs P"
        self.winner = Winner.noWin
        self.playerStart = startingPlayer
        # set in reset
        self.player1 = None
        self.player2 = None
        self.isPlayer1 = True
        self.reset()

    def get(self):
        """
        * From a player - 1 view *
        Returns dict with positions in [row, col], zero-indexed
        In API format for front end, for each player:
        player1 : { king : [2], pawns : [[2], ..., [2]], cards: [ [5x5],  [5x5]]}
        player: 1 / 2        // which player
        spare_card: [5 x 5]  // card data
        done: false / true   // when game done
        winner: 0 for none yet, 1 for p1, 2 for p2
        """
        return {**self.player1.to_dict(),
                **self.player2.to_dict(),
                "player": 1 if self.isPlayer1 else 2,
                "spare_card": self.spare_card,
                "winner": self.winner.value,
                "mode": self.mode}

    def stepApi(self, moveJson):
        move = Move(moveJson)
        return self.step(move)

    def step(self, move):
        """
        :param move:
            name: king / pawn
            id: card played
            i: (pawns only) index of pawn
            player: 1 / 2
            pos: position moved to
        :return: self.get()
        """
        curP, otherP = self.get_current_players()

        otherP.lost_pawn_last_move = False

        assert self.check_valid_move(move), \
            "\nInvalid move player " + curP.player + "\n" + str(move) + "\n" + \
            "Valid moves: {}\n".format(self.get_valid_moves(curP, self.isPlayer1))


        kingTaken = self.handle_take(otherP, move)
        newCard = self.handle_cards(curP, move)
        curP.step(move, newCard)
        if self.reached_goal(curP) or kingTaken:
            if self.verbose: print(
                "{} won: ".format(curP.player) + ("reached end" if self.reached_goal(curP) else "king taken"))
            self.winner = Winner.player1 if self.isPlayer1 else Winner.player2
            return self.get()

        self.isPlayer1 = not self.isPlayer1
        return self.get()

    def reset(self):
        p1CardsInit, p2CardsInit, [spare_card] = get_init_cards()

        self.player1 = Player(True, p1CardsInit)
        self.player2 = Player(False, p2CardsInit)
        #Can override the player start based on card stamp, if declared in game init
        if self.playerStart is None:
            self.isPlayer1 = spare_card in card_stamps[1]
        else:
            self.isPlayer1 = self.playerStart == 1 

        self.spare_card = spare_card

        # 0 for none, 1 for player 1, 2 for player 2
        self.winner = Winner.noWin

    def check_valid_move(self, move):
        assert move, "Move passed is False: {}".format(move)
        curP, otherP = self.get_current_players()
        return self.check_on_board(move) \
               and self.check_unoccupied(curP, move) \
               and self.check_move_on_card(curP, move)

    def check_on_board(self, move):
        return np.all(np.greater_equal(move.pos, 0)) and np.all(np.less_equal(move.pos, 4))

    def check_unoccupied(self, player, move):
        """
        Checks this is unoccupied by any of own pieces
        """
        if move.isKing:
            for pawn in player.pawns:
                if np.array_equal(move.pos, pawn.get()):
                    return False
        else:
            for j, pawn in enumerate(player.pawns):
                if np.array_equal(move.pos, pawn.get()) and move.i != j:
                    return False
            if np.array_equal(move.pos, player.king.get()):
                return False
        return True

    def check_move_on_card(self, player, move):
        if move.isKing:
            piecePos = player.king.get()
        else:  # pawn
            piecePos = player.pawns[move.i].get()
        posOnCard = self.board_to_card(move.pos, piecePos, self.isPlayer1)
        if np.all(np.greater_equal(posOnCard, 0)) and np.all(np.less_equal(posOnCard, 4)):
            if player.cards[move.cardId][posOnCard[0]][posOnCard[1]]:  # 1s and 0s so T/F
                return True
        if self.verbose:
            print("Move not on card {}".format(move.cardId))
            print("Move {}".format(move.pos))
            print("Piece {}".format(piecePos))
            print("Card {}".format(posOnCard))
            print(player.cards[move.cardId])
        return False

    def get_current_players(self):
        """
        :return: each player object current player (whose turn it is), and other player
        """
        current_player = self.player1 if self.isPlayer1 else self.player2
        other_player = self.player2 if self.isPlayer1 else self.player1
        return current_player, other_player

    def reached_goal(self, player):
        """
        Called post movement so check if king in goal
        """
        goalPos = [0, 2] if self.isPlayer1 else [4, 2]
        return np.array_equal(player.king.get(), goalPos)

    def handle_cards(self, curP, move):
        """
        Updates current spare card, returns new card for player
        """
        cardId = move.cardId
        card = self.spare_card
        self.spare_card = curP.cards[cardId]
        return card

    def handle_take(self, otherP, move):
        """
        Returns true if king taken -> game is won
        """
        self.handle_take_pawn(otherP, move)
        return self.check_take_king(otherP, move)

    def handle_take_pawn(self, playerOther, move):      
        for i, pawn in enumerate(playerOther.pawns):
            if np.array_equal(move.pos, pawn.get()):
                playerOther.pawns.pop(i)  # drop this pawn                
                playerOther.lost_pawn_last_move = True

    def check_take_king(self, otherP, move):
        """
         If move takes king, return True, game is won
        """
        if np.array_equal(move.pos, otherP.king.get()):
            return True
        return False

    def get_valid_moves(self, curP, isPlayer1, show_overlapping_moves=False):
        """
        :param curP: player we want the cards for
        :param isPlayer1: whether that player is p1 or p2
        :param show_overlapping_moves: when looking at opponents possible moves, we do not rule out "self captures" as
                             our piece may be moved to that position. Similary, when calculating reward we want to look
                             at the moves possible after our opponent has made theirs. Again this may appear to be a
                             "self capture" but it needs to be checked in case the opponent captures that piece.
        :return:
        """
        assert(curP.player == "player1" if isPlayer1 else "player2"), "error in get valid moves: player1 != player2"
        moves = []
        for cardId, card in enumerate(curP.cards):
            for p in np.reshape(np.where(card), [2, -1]).T:
                # king
                boardPos = self.card_to_board(curP.king.get(), p, isPlayer1)
                # since we got these moves from card we only need check they're unoccupied now and on board
                move = Move({"name": "king", "pos": boardPos, "id": cardId})
                if (self.check_unoccupied(curP, move) or show_overlapping_moves) and self.check_on_board(move):
                    moves.append(move)
                for i, pawn in enumerate(curP.pawns):
                    boardPos = self.card_to_board(pawn.get(), p, isPlayer1)
                    move = Move({"name": "pawn", "pos": boardPos, "id": cardId, "i": i})
                    if (self.check_unoccupied(curP, move) or show_overlapping_moves) and self.check_on_board(move):
                        moves.append(move)
        return moves

    def card_to_board(self, piecePos, cardPos, isPlayer1):
        """
        Returns np array (note need to convert to list for json)
        """
        if isPlayer1:
            return np.add(np.subtract(piecePos, [2, 2]), cardPos).tolist()
        else:
            return np.add(np.subtract(piecePos, cardPos), [2, 2]).tolist()

    def board_to_card(self, boardPos, piecePos, isPlayer1):
        """
        Returns np array (note need to convert to list for json)
        """
        if isPlayer1:
            return np.subtract(np.add(boardPos, [2, 2]), piecePos).tolist()
        else:
            return np.subtract(np.add([2, 2], piecePos), boardPos).tolist()


class PvBot(PvP):
    def __init__(self, agent, *args, **kwargs):
        super(PvBot, self).__init__(*args, **kwargs)
        self.agent = agent
        self.mode = "P vs Bot"

    def stepApi(self, moveJson):
        """
        API call for step()
        """
        move = Move(moveJson)
        state = self.step(move)
        return state

    def step(self, move):
        """
        Steps player only (stepBot for bot)
        """
        state = super(PvBot, self).step(move)
        return state

    def stepBot(self):
        """
        Steps the bot
        """
        state = self.get()
        if self.winner is Winner.noWin:
            agentMove = self.agent.get_action(self)
            state = super(PvBot, self).step(agentMove)
        return state


class BotVsBot(PvP):
    def __init__(self, agent1, agent2, *args, **kwargs):
        super(BotVsBot, self).__init__(*args, **kwargs)
        self.agent1 = agent1
        self.agent2 = agent2
        self.mode = "Bot vs Bot"

    def step(self, move):
        """
        Does nothing here
        """
        return self.get()

    def stepBot(self):
        """
        Steps the bot
        """
        agent = self.agent1 if self.isPlayer1 else self.agent2
        agentMove = agent.get_action(self)
        state = super(BotVsBot, self).step(agentMove)
        return state
