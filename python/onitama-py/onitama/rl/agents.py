import numpy as np
import random


class RandomAgent:
    def __init__(self, seed, isPlayer1=False):
        """
        Assumes player 2 as this is normal
        """
        self.isPlayer1 = isPlayer1
        print(seed, type(seed))
        np.random.seed(seed)

    def get_action(self, state):
        """
        State is a game object eg. PvP
        """
        player = state.player1 if self.isPlayer1 else state.player2
        ac = np.random.choice(state.get_valid_moves(player, self.isPlayer1))
        return ac

class SimpleAgent:
    '''
    Simple agent objectives (in order of priority).
    1) Move king to enemy home or capture enemy king
    2) If own king is attackable: i) If 1 attacker: Capture attacker with pawn
                                  ii) Safe capture with king
                                  iii) Retreat king
    3) If own pawn is attackable: i) Safe capture with attacked pawn
                                  ii) Safe capture with other pawn
                                  iii) Unsafe capture with attacked pawn
                                  iv) Retreat attacked pawn
    3) Attempts to capture an enemy pawn without it being attackable
    4) Attempts to move a random piece without it being attackable
    5) Attempts to move a random piece
    '''

    def __init__(self, seed, isPlayer1=False, isVerbose=False, threshold=0.):
        self.isPlayer1 = isPlayer1
        self.isVerbose = isVerbose
        self.threshold = threshold
        print(seed, type(seed))
        random.seed(seed)
        np.random.seed(seed)

    def get_action(self, state):
        agent_player = state.player1 if self.isPlayer1 else state.player2
        opp_player = state.player2 if self.isPlayer1 else state.player1

        all_moves = state.get_valid_moves(agent_player, self.isPlayer1)
        opp_moves = state.get_valid_moves(opp_player, not self.isPlayer1, show_overlapping_moves=True)

        # Agent king, opponent king, and opponent pawn positions
        king = agent_player.king  # [r,c]
        pawns = agent_player.pawns  # [[r,c]]
        opp_king = opp_player.king  # [r,c]
        opp_pawns = opp_player.pawns  # [[r,c]]

        # Safe moves are moves which end up on a square not attackable by an opponents piece
        safe_moves = [move for move in all_moves if move.pos not in [opp_move.pos for opp_move in opp_moves]]
        # Captures are moves where an opponent's pawn is captured
        captures = [move for move in all_moves if move.pos in [opp_pawn.pos for opp_pawn in opp_pawns]]
        # Safe captures are safe moves where a pawn is captured
        safe_captures = [move for move in safe_moves if move in captures]

        if self.isVerbose:
            print("all_moves")
            for move in all_moves:
                print(move)
            print("opp_moves")
            for move in opp_moves:
                print(move)

        #For threshold fraction of time, choose random move
        randomChoice = random.uniform(0,1)
        if randomChoice < self.threshold:
            return np.random.choice(all_moves)
        
        # Winning moves
        enemy_shrine_pos = [0, 2] if self.isPlayer1 else [4, 2]
        winning_moves = [move for move in all_moves if move.isKing and move.pos == enemy_shrine_pos
                         or move.pos == opp_king.pos]
        # 1) Agent tries to win
        if len(winning_moves) > 0:
            if self.isVerbose:
                print("winning move")
            return winning_moves[0]

        # 2) King is under attack
        opp_a_moves = [opp_a_move for opp_a_move in opp_moves if opp_a_move.pos == king.pos]
        opp_a_piece_ids = [opp_a_move.i for opp_a_move in opp_a_moves]
        opp_a_piece_pos = [opp_a_pawn.pos for opp_a_pawn in opp_pawns if opp_a_pawn.id in opp_a_piece_ids]
        if len(opp_a_moves) > 0:
            if len(opp_a_moves) == 1:
                # Only 1 attacker.
                # i) Attempts to capture attacker with pawn
                if self.isVerbose:
                    print("king attacked by 1")
                    for move in opp_a_moves:
                        print(move)
                saving_pawn_captures = [move for move in all_moves if
                                        move.pos in opp_a_piece_pos and not move.isKing]
                if len(saving_pawn_captures) > 0:
                    if self.isVerbose:
                        print("saving pawn capture")
                        for move in saving_pawn_captures:
                            print(move)
                    return np.random.choice(saving_pawn_captures)
            else:
                if self.isVerbose:
                    print("king attacked by >1")

            # ii) Attempts to safe capture a pawn with king
            king_captures = [move for move in safe_captures if move.isKing]
            if len(king_captures) > 0:
                if self.isVerbose:
                    print("escaping king safe capture")
                    for move in king_captures:
                        print(move)
                return np.random.choice(king_captures)

            # iii) King retreats to a safe square
            if len(opp_a_piece_ids) > 0:
                retreating_king_moves = [move for move in safe_moves if move.isKing]
                if len(retreating_king_moves) > 0:
                    if self.isVerbose:
                        print("retreating king move")
                        for move in retreating_king_moves:
                            print(move)
                    return np.random.choice(retreating_king_moves)
                else:
                    if self.isVerbose:
                        print("no retreats possible")

        # 3) One of our pawns is under attack
        threatened_pawns_id = [pawn.id for pawn in pawns if pawn.pos in
                               [opp_move.pos for opp_move in opp_moves]]
        threatened_pawns = [pawn for pawn in pawns if pawn.pos in
                            [opp_move.pos for opp_move in opp_moves]]
        if len(threatened_pawns_id) > 0:
            if self.isVerbose:
                print("pawn(s) threatened")
                for pawn in threatened_pawns:
                    print("pos:", pawn.pos, "id:", pawn.id)

            # i) Safe capture with attacked pawn
            safe_evading_captures = [safe_capture for safe_capture in safe_captures
                                     if safe_capture.i in threatened_pawns_id]
            if len(safe_evading_captures) > 0:
                if self.isVerbose:
                    print("safe evading capture")
                    for move in safe_evading_captures:
                        print(move)
                return np.random.choice(safe_evading_captures)

            # ii) Safe capture with other pawn
            if len(safe_captures) > 0:
                if self.isVerbose:
                    print("safe capture")
                    for move in safe_captures:
                        print(move)
                return np.random.choice(safe_captures)

            # iii) Unsafe capture with attacked pawn
            threatened_pawn_captures = [move for move in all_moves if
                                        move.i in threatened_pawns_id and move in captures]
            if len(threatened_pawn_captures) > 0:
                if self.isVerbose:
                    print("unsafe capture")
                    for move in threatened_pawn_captures:
                        print(move)
                return np.random.choice(threatened_pawn_captures)

            # iv) Retreat attacked pawn
            threatened_pawn_retreats = [move for move in safe_moves if move.i in threatened_pawns_id]
            if len(threatened_pawn_retreats) > 0:
                if self.isVerbose:
                    print("pawn retreated")
                    for move in threatened_pawn_retreats:
                        print(move)
                return np.random.choice(threatened_pawn_retreats)

        # 4) Capture an undefended enemy piece
        if len(safe_captures) > 0:
            if self.isVerbose:
                print("safe capture")
                for move in safe_captures:
                    print(move)
            return np.random.choice(safe_captures)

        # 5) Chooses a random safe move
        # i) Moves pawns to the central 9 squares of the board.
        central_safe_pawn_moves = [move for move in safe_moves if move.isKing is False and
                                   move.pos in [[1, 1], [1, 2], [1, 3],
                                                [2, 1], [2, 2], [2, 3],
                                                [3, 1], [3, 2], [3, 3], ]
                                   ]
        if len(central_safe_pawn_moves) > 0:
            move = np.random.choice(central_safe_pawn_moves)
            if self.isVerbose:
                print("random safe central pawn move")
                print(move)
            return move
        # ii) Chooses any other safe move
        if len(safe_moves) > 0:
            move = np.random.choice(safe_moves)
            if self.isVerbose:
                print("random safe move")
                print(move)
            return move

        # 6) All moves left are unsafe. Prefers unsafe pawn moves.
        else:
            unsafe_pawn_moves = [move for move in all_moves if move.isKing is False]
            if len(unsafe_pawn_moves) > 0:
                if self.isVerbose:
                    print("random unsafe pawn move")
                    for move in unsafe_pawn_moves:
                        print(move)
                return np.random.choice(unsafe_pawn_moves)
            else:
                if self.isVerbose:
                    print("random unsafe king move")
                    for move in all_moves:
                        print(move)
                return np.random.choice(all_moves)



    '''
    Simple Agent Problems:

    safe_moves: In the following setting where dashes are empty spaces, X are the
    random agent's pawns and O is an enemy pawn.
    Imagine both Xs can attack diagonally and the O can attack forwards. All pieces could move
    to the central square.
    The move by X to the middle square should be safe because our piece can recapture, but won't
    be in our list.
    [-, -, -, -, -]
    [-, -, O, -, -]
    [-, -, -, -, -]
    [-, X, -, X, -]
    [-, -, -, -, -]

    no attempt made to avoid enemy king winning by reaching our home
    '''