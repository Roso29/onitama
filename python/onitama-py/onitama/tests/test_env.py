from onitama.rl import OnitamaEnv, OnitamaSelfPlayEnv, actionToMove, moveToMask, RandomAgent, DQNMaskedCNNPolicy
from stable_baselines.deepq import DQN
from onitama.game import Move, get_move, PvBot
from onitama.game.cards import only_sideways
from onitama.rl.env import flip_game_view
import numpy as np
import unittest


class EnvTest(unittest.TestCase):
    seed = 123124

    def testMoveForwardsPlayer2(self):
        # Test move_forwards
        ##Move player 2 pawn (id=0) forward 2 squares
        ##Reward should = 0.2 (2 moves * 0.1)
        env = OnitamaEnv(self.seed, isPlayer1=True)
    
        moveJson = {"pos": [2, 0],
                    "name": "pawn",
                    "i": 0,
                    "id": 0}
    
        move = Move(moveJson)
        env.game.player1.step(move, None)
        self.assertEqual(env.get_reward(), 0.01)
    #
    # def testPlayer2TakePawn(self):
    #     # Test Player1 taking Player2s pawn at location [0,0]
    #     # NOTE: This test will only pass if the move is NOT checked to be a valid move
    #     env = OnitamaEnv(self.seed)
    #     moveJson = {"pos": [0, 0],
    #                 "name": "pawn",
    #                 "i": 0,
    #                 "id": 0}
    #     move = Move(moveJson)
    #
    #     env.game.step(move)
    #
    #     self.assertEqual(env.get_reward(), 0.1+4*0.01)

    def test_error(self):
        env = OnitamaEnv(self.seed)
        obs = env.reset()

    def test_move_to_mask(self):
        env = OnitamaEnv(self.seed)
        ac = np.zeros((5, 5, 50))
        # ac has to be a piece
        # note all pieces in orig positions, for p1 it's [4, *], king at [4, 2]
        ac[4, 2, 29] = 1
        move = actionToMove([i[0] for i in np.where(ac)], env.game, env.isPlayer1, env.mask_shape)
        mask = moveToMask(move, env.game.player1)
        assert np.all([a[0] == m for a, m in zip(np.where(ac), mask)]), "Ac : {}\nMask : {}".format(np.where(ac), mask)

    def test_move_to_mask_pawn(self):
        env = OnitamaEnv(self.seed)
        ac = np.zeros((5, 5, 50))
        # ac has to be a piece
        # note all pieces in orig positions, for p1 it's [4, *], king at [4, 2]
        ac[4, 1, 29] = 1
        move = actionToMove([i[0] for i in np.where(ac)], env.game, env.isPlayer1, env.mask_shape)
        mask = moveToMask(move, env.game.player1)
        assert np.all([a[0] == m for a, m in zip(np.where(ac), mask)]), "Ac : {}\nMask : {}".format(np.where(ac), mask)

    def test_mask_to_move(self):
        env = OnitamaEnv(self.seed)
        # note all pieces in orig positions, for p1 it's [4, *], king at [4, 2]
        move = get_move([1, 1], True, 0, -1)
        mask = moveToMask(move, env.game.player1)
        ac = np.zeros((5, 5, 50))
        ac[mask] = 1
        move2 = actionToMove([i[0] for i in np.where(ac)], env.game, env.isPlayer1, env.mask_shape)
        assert move.pos == move2.pos, "pos Orig : {}\nNew : {}".format(move, move2)
        assert move.isKing == move2.isKing, "isKing Orig : {}\nNew : {}".format(move, move2)
        assert move.i == move2.i, "i Orig : {}\nNew : {}".format(move, move2)
        assert move.cardId == move2.cardId, "CardID Orig : {}\nNew : {}".format(move, move2)

    def test_mask_to_move_pawn(self):
        env = OnitamaEnv(self.seed)
        move = get_move([1, 1], False, 0, 1)
        mask = moveToMask(move, env.game.player1)
        ac = np.zeros((5, 5, 50))
        ac[mask] = 1
        move2 = actionToMove([i[0] for i in np.where(ac)], env.game, env.isPlayer1, env.mask_shape)
        assert move.pos == move2.pos, "pos Orig : {}\nNew : {}".format(move, move2)
        assert move.isKing == move2.isKing, "isKing Orig : {}\nNew : {}".format(move, move2)
        assert move.i == move2.i, "i Orig : {}\nNew : {}".format(move, move2)
        assert move.cardId == move2.cardId, "CardID Orig : {}\nNew : {}".format(move, move2)

    def test_valid_moves(self):
        env = OnitamaEnv(self.seed, agent_type=RandomAgent)
        env.reset()
        assert env.game.isPlayer1, "Wrong player value"
        for move in env.game.get_valid_moves(env.game.player1, True):
            assert env.game.check_valid_move(move), "Before found incorrect valid move {} \n in p1 {}".format(
                move, env.game.player1)
        valid_moves = env.game.get_valid_moves(env.game.player1, True)
        env.game.step(valid_moves[0])
        assert not env.game.isPlayer1, "Wrong player value"
        for move in env.game.get_valid_moves(env.game.player2, False):
            assert env.game.check_valid_move(move), "After found incorrect valid move in p2 {}".format(move)
        env.game.stepBot()
        for move in env.game.get_valid_moves(env.game.player1, True):
            assert env.game.check_valid_move(move), "After found incorrect valid move in p1 {}".format(move)

    def test_env_random_agent(self):
        """
        Would expect 50 / 50 of random unmasked actions compared to random agent
        """
        env = OnitamaEnv(self.seed, agent_type=RandomAgent)
        n_episodes = 10
        wins = 0
        for ep in range(n_episodes):
            obs = env.reset()
            mask = obs[:, :, 9:]
            done = False
            while not done:
                valid_acs = [np.ravel_multi_index(ac, (5, 5, 50)) for ac in zip(*np.where(mask))]
                action = env.action_space.sample()
                while not action in valid_acs:
                    action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                mask = obs[:, :, 9:]
                if done:
                    if info["winner"] == 1:
                        wins += 1
        print("Won {} of {}".format(wins, n_episodes))

    def test_game_flip(self):
        """
        Not really a test just prints out values to manually check but I put it here
        """
        game = PvBot(RandomAgent(self.seed), self.seed)
        game_flip = flip_game_view(game)
        game_flip_flip = flip_game_view(game_flip)
        print(game.player1)
        print(game_flip_flip.player1)
        print(game_flip.player1)
        print(game.player2)
        print(game_flip_flip.player2)
        print(game_flip.player2)

    def test_self_play(self):
        """
        Make sure it all runs, also might expect fairly evenly matched at the start?
        """
        env = OnitamaSelfPlayEnv(self.seed)
        p1 = DQN(DQNMaskedCNNPolicy, env, learning_starts=10)
        p2 = DQN(DQNMaskedCNNPolicy, env, learning_starts=10)
        env.setSelfPlayModel(p1)
        deterministic = False
        n_episodes = 10
        wins = 0
        for ep in range(n_episodes):
            ob = env.reset()
            done = False
            while not done:
                # print("Player 1")
                ac, _ = p1.predict(ob, deterministic=deterministic)
                # print(ac)
                ob, _, done, info = env.step(ac)
                # print("Player 2")
                ac, _ = p2.predict(ob, deterministic=deterministic)
                # print(ac)
                ob, _, done, info = env.step(ac)
                if done:
                    if info["winner"] == 1:
                        wins += 1
        print("P1 won {} of {}".format(wins, n_episodes))

    def test_no_valid_moves(self):
        game = PvBot(RandomAgent, self.seed)
        game.player1.cards = [only_sideways, only_sideways]
        game.player2.cards = [only_sideways, only_sideways]
        game.spare_card = only_sideways
        # TODO


if __name__ == "__main__":
    np.random.seed(111)
    unittest.main()
