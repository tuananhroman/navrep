from datetime import datetime
import os

#from stable_baselines import PPO2
#from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

from navrep.tools.custom_policy import Custom1DPolicy, ARCH, _C
from navrep.envs.e2eenv import E2E1DNavRepEnv
from navrep.tools.sb_eval_callback import NavrepEvalCallback
from navrep.tools.commonargs import parse_common_args

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from navrep.scripts.custom_policy_sb3 import CustomMlp

if __name__ == "__main__":
    args, _ = parse_common_args()

    DIR = os.path.expanduser("~/navrep/models/gym")
    LOGDIR = os.path.expanduser("~/navrep/logs/gym")
    if args.dry_run:
        DIR = "/tmp/navrep/models/gym"
        LOGDIR = "/tmp/navrep/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    CONTROLLER_ARCH = "_{}_C{}".format(ARCH, _C)
    LOGNAME = "e2e1dnavreptrainenv_" + START_TIME + "_PPO" + "_E2E1D" + CONTROLLER_ARCH
    LOGPATH = os.path.join(LOGDIR, LOGNAME + ".csv")
    MODELPATH = os.path.join(DIR, LOGNAME + "_ckpt")
    MODELPATH2 = os.path.join(DIR, "e2e1dnavreptrainenv_latest_PPO_ckpt")
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)

    MILLION = 1000000
    TRAIN_STEPS = args.n
    if TRAIN_STEPS is None:
        TRAIN_STEPS = 60 * MILLION

    N_ENVS = 1
    if args.debug:
        env = DummyVecEnv([lambda: E2E1DNavRepEnv(silent=True, scenario='train')]*N_ENVS)
    else:
        env = SubprocVecEnv([lambda: E2E1DNavRepEnv(silent=True, scenario='train')]*N_ENVS,
                            start_method='spawn')
    eval_env = E2E1DNavRepEnv(silent=True, scenario='train')
    def test_env_fn():  # noqa
        return E2E1DNavRepEnv(silent=True, scenario='test')
    cb = NavrepEvalCallback(eval_env, test_env_fn=test_env_fn,
                            logpath=LOGPATH, savepath=MODELPATH, verbose=1)
    model = PPO(CustomMlp, env, verbose=0)
    model.learn(total_timesteps=TRAIN_STEPS+1, callback=cb)
    obs = env.reset()

    model.save(MODELPATH)
    model.save(MODELPATH2)
    print("Model '{}' saved".format(MODELPATH))

    del model

    model = PPO2.load(MODELPATH)

    env = E2E1DNavRepEnv(silent=True, scenario='train')
    obs = env.reset()
    for i in range(512):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            env.reset()
#         env.render()
