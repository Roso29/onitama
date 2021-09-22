# Third party
from gym.envs.registration import register

register(id="Onitama-v0", entry_point="onitama.rl.env:OnitamaEnv")
register(id="OnitamaSelfPlay-v0", entry_point="onitama.rl.env:OnitamaSelfPlayEnv")