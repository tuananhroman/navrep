from datetime import datetime
import os
import torch as th

from navrep.scripts.custom_policy_sb3 import CustomMlpPolicy
from navrep.envs.e2eenv import E2E1DNavRepEnv
from navrep.tools.sb_eval_callback import NavrepEvalCallback
from navrep.tools.commonargs import parse_common_mlp_args

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

###HYPERPARAMETER###
gamma = 0.99
n_steps = 128
ent_coef = 0.01
learning_rate = 2.5e-4
vf_coef = 0.5
max_grad_norm = 0.5
lam = 0.95
nminibatches = 4
noptepochs = 4
cliprange = 0.2
####################


def parse_string(string: str):
    string_arr = string.split("-")
    int_list = []
    for string in string_arr:
        int_list.append(int(string))
    return int_list


def get_net_arch(args: parse_common_mlp_args()):
    body = parse_string(args.body)
    policy = parse_string(args.pi)
    value = parse_string(args.vf)
    return body + [dict(vf=value, pi=policy)]


def get_act_fn(act_fn_string: str):
    if act_fn_string == "relu":
        return th.nn.ReLU
    elif act_fn_string == "sigmoid":
        return th.nn.Sigmoid
    elif act_fn_string == "tanh":
        return th.nn.Tanh


if __name__ == "__main__":
    args, _ = parse_common_mlp_args()

    DIR = os.path.expanduser("~/navrep/models/gym")
    LOGDIR = os.path.expanduser("~/navrep/logs/gym")
    if args.dry_run:
        DIR = "/tmp/navrep/models/gym"
        LOGDIR = "/tmp/navrep/logs/gym"
    START_TIME = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    if args.custom:
        CONTROLLER_ARCH = "_B"+args.body+"_V"+args.vf+"_P"+args.pi
    else:
        CONTROLLER_ARCH = "_MLP_ARENA2D"
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
        env = DummyVecEnv([lambda: E2E1DNavRepEnv(silent=True, scenario='train')] * N_ENVS)
    else:
        env = SubprocVecEnv([lambda: E2E1DNavRepEnv(silent=True, scenario='train')] * N_ENVS,
                            start_method='spawn')
    eval_env = E2E1DNavRepEnv(silent=True, scenario='train')


    def test_env_fn():  # noqa
        return E2E1DNavRepEnv(silent=True, scenario='test')


    cb = NavrepEvalCallback(eval_env, test_env_fn=test_env_fn,
                            logpath=LOGPATH, savepath=MODELPATH, verbose=1)

    if args.custom:
        net_arch = get_net_arch(args)
        act_fn = get_act_fn(args.act_fn)
        print("MLP Architecture: \n"
              "Body: {}  -  Value:  {}  -  Policy:  {}  \n"
              "Activation Function: {}".format(net_arch[:-1], net_arch[-1]["vf"], net_arch[-1]["pi"], args.act_fn))
        model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=net_arch, activation_fn=act_fn),
                    verbose=0, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef,
                    learning_rate=learning_rate, vf_coef=vf_coef, max_grad_norm=max_grad_norm, gae_lambda=lam,
                    batch_size=nminibatches, n_epochs=noptepochs, clip_range=cliprange)
    else:
        model = PPO(CustomMlpPolicy, env, verbose=0, gamma=gamma, n_steps=n_steps, ent_coef=ent_coef,
                    learning_rate=learning_rate, vf_coef=vf_coef, max_grad_norm=max_grad_norm, gae_lambda=lam,
                    batch_size=nminibatches, n_epochs=noptepochs, clip_range=cliprange)
    model.learn(total_timesteps=TRAIN_STEPS + 1, callback=cb)
    obs = env.reset()

    model.save(MODELPATH)
    model.save(MODELPATH2)
    print("Model '{}' saved".format(MODELPATH))

    del model

    model = PPO.load(MODELPATH)

    env = E2E1DNavRepEnv(silent=True, scenario='train')
    obs = env.reset()
    for i in range(512):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            env.reset()
#         env.render()
