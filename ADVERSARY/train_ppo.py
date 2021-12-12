"""
    This file is the executable for running PPO. It is based on this medium article: 
    https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""
import sys
import os
import json
from comet_ml import Experiment
import torch
import argparse

from lightsim2grid import LightSimBackend
import grid2op
from grid2op.Action import PlayableAction, TopologyChangeAndDispatchAction
from grid2op.Reward import (
    CombinedScaledReward,
    L2RPNSandBoxScore,
    L2RPNReward,
    GameplayReward,
)

# from kaist_agent.Kaist import Kaist
from CONTROLLER.agent import Agent

from ppo.ppo import PPO
from ppo.nnpytorch import FFN
from ppo.gat_ppo import GPPO


def train(
    env, agent, state_mean, state_std, hyperparameters, actor_model, critic_model, adv_name
):
    """
    Trains the model.
    Parameters:
        env - the environment to train on
        hyperparameters - a dict of hyperparameters to use, defined in main
        actor_model - the actor model to load in if we want to continue training
        critic_model - the critic model to load in if we want to continue training
    Return:
        None
    """
    print(f"Training", flush=True)

    experiment = Experiment(project_name="285-fp", api_key=os.getenv("COMET_API_KEY"))
    experiment.set_name(adv_name)

    # Create a model for PPO.
    # ! when using our own architecture CHANGE THIS
    # model = PPO(
    #     experiment=experiment,
    #     env=env,
    #     agent=agent,
    #     policy_class=FFN,
    #     state_mean=state_mean,
    #     state_std=state_std,
    #     name=adv_name,
    #     **hyperparameters,
    # )

    model = GPPO(
        experiment=experiment,
        env=env,
        agent=agent,
        policy_class=None,
        state_mean=state_mean,
        state_std=state_std,
        name=adv_name,
        **hyperparameters,
    )

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != "" and critic_model != "":
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif (
        actor_model != "" or critic_model != ""
    ):  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!"
        )
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)


def get_args():
    """
    Description:
    Parses arguments at command line.
    Parameters:
        None
    Return:
        args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", dest="mode", type=str, default="train"
    )  # can be 'train' or 'test'
    parser.add_argument(
        "--actor_model", dest="actor_model", type=str, default=""
    )  # your actor model filename
    parser.add_argument(
        "--critic_model", dest="critic_model", type=str, default=""
    )  # your critic model filename

    parser.add_argument("-data", "--datapath", type=str, default="./data")
    parser.add_argument(
        "-c", "--case", type=str, default="wcci", choices=["sand", "wcci"]
    )
    parser.add_argument("--controller", type=str, default="./result/wcci_run_0/model/")
    parser.add_argument("--c_suffix", type=str, default="last")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--adv_name", type=str, default="")

    parser.add_argument(
        "-ap",
        "--attack_period",
        type=int,
        default=50,
        help="frequency of opponent attack",
    )

    args = parser.parse_args()

    return args


def main(args):
    """
    The main function to run.
    Parameters:
        args - the arguments parsed from command line
    Return:
        None
    """
    # NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
    # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
    # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters

    # Environment
    DATA_DIR = args.datapath
    case = {"wcci": "l2rpn_wcci_2020", "sand": "l2rpn_case14_sandbox"}
    backend = LightSimBackend()
    env_name = os.path.join(args.datapath, case[args.case])
    env = grid2op.make(env_name, reward_class=CombinedScaledReward, backend=backend)

    # Agent
    agent_name = "kaist"
    # data_dir = os.path.join("kaist_agent/data")
    with open(os.path.join(os.path.join(args.controller, "../"), "param.json"), "r", encoding="utf-8") as f:
        param = json.load(f)
    param["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(param)
    og_smac = os.path.join(DATA_DIR, case[args.case])
    state_mean = torch.load(
        os.path.join(og_smac, "mean.pt"), map_location=param["device"]
    ).cpu()
    state_std = torch.load(
        os.path.join(og_smac, "std.pt"), map_location=param["device"]
    ).cpu()
    state_std = state_std.masked_fill(state_std < 1e-5, 1.0)
    state_mean[0, sum(env.observation_space.shape[:20]) :] = 0
    state_std[0, sum(env.observation_space.shape[:20]) :] = 1
    agent = Agent(experiment=None, env=env, **param)
    agent.load_mean_std(state_mean, state_std)
    agent.sim_trial = 0
    agent.load_model(args.controller, name=args.c_suffix)

    hyperparameters = {
        "seed": args.seed,
        "timesteps_per_batch": 448, # 2048
        "max_timesteps_per_episode": 200, # 200
        "gamma": 0.99,
        "n_updates_per_iteration": 10,
        "lr": 3e-4,
        "clip": 0.2,
        "lines_attacked": [
            "10_11_11",
            "12_16_20",
            "16_17_22",
            "16_18_23",
            "16_21_27",
            "16_21_28",
            "16_33_48",
            "14_35_53",
            "9_16_18",
            "9_16_19",
        ],
        "attack_duration": 10,
        "attack_period": args.attack_period,
        "danger": 0.9,
    }

    # Train or test, depending on the mode specified
    train(
        env=env,
        agent=agent,
        state_mean=state_mean,
        state_std=state_std,
        hyperparameters=hyperparameters,
        actor_model=args.actor_model,
        critic_model=args.critic_model,
        adv_name=args.adv_name,
    )


if __name__ == "__main__":
    args = get_args()  # Parse arguments from command line
    main(args)