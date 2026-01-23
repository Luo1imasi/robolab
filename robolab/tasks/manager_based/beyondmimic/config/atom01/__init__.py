
import gymnasium as gym

from . import agents, env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Atom01-BeyondMimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Atom01BeyondMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.agent_cfg:Atom01BeyondMimicPPORunnerCfg",
    },
)
