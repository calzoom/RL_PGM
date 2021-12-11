import os
import csv
import json
import random
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import torch
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from custom_reward import *
from agent import Agent
from train import TrainAgent

# from matplotlib import pyplot as plt
# import matplotlib.cbook
# import warnings
# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# from simple_opponents.random_opponent import RandomOpponent, WeightedRandomOpponent
from ADVERSARY.ppo.ppo import PPO

# from PPO.nnpytorch import FFN

# from track1.agent import Track1PowerNetAgent
# from kaist_agent_2.Kaist import Kaist
# from nanyang_agent.agents import MyAgent as Nanyang
# from d3qn_agent.DoubleDuelingDQN import DoubleDuelingDQN as D3QNAgent


ENV_CASE = {
    "5": "rte_case5_example",
    "sand": "l2rpn_case14_sandbox",
    "wcci": "l2rpn_wcci_2020",
    "parl": "l2rpn_neurips_2020_track1_small",
}

DATA_SPLIT = {
    "5": ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    "sand": (
        list(range(0, 40 * 26, 40)),
        list(range(1, 100 * 10 + 1, 100)),
        list(range(2, 100 * 10 + 2, 100)),
    ),
    "wcci": (
        [
            17,
            240,
            494,
            737,
            990,
            1452,
            1717,
            1942,
            2204,
            2403,
            19,
            242,
            496,
            739,
            992,
            1454,
            1719,
            1944,
            2206,
            2405,
            230,
            301,
            704,
            952,
            1008,
            1306,
            1550,
            1751,
            2110,
            2341,
            2443,
            2689,
        ],
        list(range(2880, 2890)),
        [18, 241, 495, 738, 991, 1453, 1718, 1943, 2205, 2404],
    ),
    "parl": ([], [], [18, 241, 495, 738, 991, 1453, 1718, 1943, 2205, 2404]),
}

MAX_FFW = {"5": 5, "sand": 26, "wcci": 26, "parl": 26}


def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--case", type=str, default="wcci", choices=["sand", "wcci", "5", "parl"]
    )
    parser.add_argument("-gpu", "--gpuid", type=int, default=0)
    parser.add_argument("-data", "--datapath", type=str, default="./data")
    parser.add_argument("-out", "--output", type=str, default="./result")

    parser.add_argument(
        "-hn",
        "--head_number",
        type=int,
        default=8,
        help="the number of head for attention",
    )
    parser.add_argument(
        "-sd",
        "--state_dim",
        type=int,
        default=128,
        help="dimension of hidden state for GNN",
    )
    parser.add_argument(
        "-nh", "--n_history", type=int, default=6, help="length of frame stack"
    )
    parser.add_argument("-do", "--dropout", type=float, default=0.0)

    parser.add_argument(
        "-r",
        "--rule",
        type=str,
        default="c",
        choices=["c", "d", "o", "f"],
        help="low-level rule (capa, desc, opti, fixed)",
    )
    parser.add_argument(
        "-thr",
        "--threshold",
        type=float,
        default=0.1,
        help="[-1, thr) -> bus 1 / [thr, 1] -> bus 2",
    )
    parser.add_argument(
        "-dg",
        "--danger",
        type=float,
        default=0.9,
        help="the powerline with rho over danger is regarded as hazardous",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=int,
        default=5,
        help='this agent manages the substations containing topology elements over "mask"',
    )
    parser.add_argument("-mhi", "--mask_hi", type=int, default=19)
    parser.add_argument("-mll", "--max_low_len", type=int, default=19)

    parser.add_argument("-l", "--last", action="store_true")
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("--opp", type=str)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args()
    return args


def read_ffw_json(path, chronics, case):
    res = {}
    for i in chronics:
        for j in range(MAX_FFW[case]):
            with open(
                os.path.join(path, "%d_%d.json" % (i, j)), "r", encoding="utf-8"
            ) as f:
                a = json.load(f)
                res[(i, j)] = (
                    a["dn_played"],
                    a["donothing_reward"],
                    a["donothing_nodisc_reward"],
                )
            if i >= 2880:
                break
    return res


def read_loss_json(path, chronics):
    losses = {}
    loads = {}
    chronics = list(set(chronics))
    for i in chronics:
        json_path = os.path.join(path, "%s.json" % i)
        with open(json_path, "r", encoding="utf-8") as f:
            res = json.load(f)
        losses[i] = res["losses"]
        loads[i] = res["sum_loads"]
    return losses, loads


if __name__ == "__main__":
    args = cli()

    # settings
    model_name = args.name  # + "_" + args.opp + "_adv_" + str(args.seed)
    print("model_name: ", model_name)

    OUTPUT_DIR = "../result"
    DATA_DIR = "../data"
    output_result_dir = os.path.join(OUTPUT_DIR, model_name)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = ENV_CASE[args.case]
    env_path = os.path.join(DATA_DIR, env_name)
    chronics_path = os.path.join(env_path, "chronics")
    train_chronics, valid_chronics, test_chronics = DATA_SPLIT[args.case]
    dn_json_path = os.path.join(env_path, "json")

    # select chronics
    dn_ffw = read_ffw_json(dn_json_path, test_chronics, args.case)

    # when small number of train, test chronic
    ep_infos = {}
    if os.path.exists(dn_json_path):
        for i in list(set(test_chronics)):
            with open(
                os.path.join(dn_json_path, f"{i}.json"), "r", encoding="utf-8"
            ) as f:
                ep_infos[i] = json.load(f)

    mode = "last" if args.last else "best"

    env = grid2op.make(
        env_path,
        test=True,
        reward_class=L2RPNSandBoxScore,
        backend=LightSimBackend(),
        other_rewards={"loss": LossReward},
    )
    # env.deactivate_forecast()
    env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    env.parameters.NB_TIMESTEP_RECONNECTION = 12
    env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
    print(env.parameters.__dict__)

    # Nanyang Agent
    # my_agent = Nanyang(env.action_space)

    # # D3QN Agent
    # my_agent = D3QNAgent(env.observation_space,
    #                 env.action_space,
    #                 is_training=False)
    # print(agent.Qmain.get_weights())
    # Load weights from file
    # my_agent.load("d3qn_agent/models/d3qn_parl.h5")

    # # KAIST Agent
    model_path = os.path.join(output_result_dir, "model")
    my_agent = Agent(env, **vars(args))
    mean = torch.load(os.path.join(env_path, "mean.pt"))
    std = torch.load(os.path.join(env_path, "std.pt"))
    my_agent.load_mean_std(mean, std)
    my_agent.load_model(model_path, mode)

    EASY_LINES = [
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
    ]
    SAND_LINES = ["1_2_2", "1_4_4", "2_3_5", "6_8_19", "8_13_11", "9_10_12"]

    attack_period = 50
    attack_duration = 10
    attack_lines = EASY_LINES

    hyperparameters = {
        "lines_attacked": attack_lines,
        "attack_duration": attack_duration,
        "attack_period": attack_period,
        "danger": 0.9,
        "state_dim": 1062,  # 342 for sandbox
    }

    data_dir = os.path.join("../data/l2rpn_wcci_2020")
    with open(
        os.path.join("../result/wcci_run_0/param.json"), "r", encoding="utf-8"
    ) as f:
        param = json.load(f)
    # print(param)
    mean = torch.load(os.path.join(data_dir, "mean.pt"), map_location="cpu").cpu()
    std = torch.load(os.path.join(data_dir, "std.pt"), map_location="cpu").cpu()

    for i in range(1):
        ##opponent_ppo.actor.load_state_dict(torch.load('./ppo_actor_kaist.pth'))
        # opponent_ppo_d3qn = PPO(env=env, agent=my_agent, policy_class=FFN, state_mean=mean, state_std=std, **hyperparameters)
        # opponent_ppo_d3qn.actor.load_state_dict(torch.load('../ppo_actor_easy_0.02clip.pth'))
        # opponent_ppo_kaist = PPO(env=env, agent=my_agent, policy_class=FFN, state_mean=mean, state_std=std, **hyperparameters)
        # opponent_ppo_kaist.actor.load_state_dict(torch.load(f'../ppo_actor_nanyang_wcci.pth'))
        # opponent_ppo_nanyang = PPO(env=env, agent=my_agent, policy_class=FFN, state_mean=None, state_std=None, transfer=True, **hyperparameters2)
        # opponent_ppo_nanyang.actor.load_state_dict(torch.load('../ppo_actor_wcci_transfer_0.02clip.pth'))

        # opponent_ppo_kaist = PPO(
        #     env=env,
        #     agent=my_agent,
        #     policy_class=FFN,
        #     state_mean=mean,
        #     state_std=std,
        #     **hyperparameters,
        # )
        # opponent_ppo_kaist.actor.load_state_dict(
        #     torch.load("../ppo_actor_kaist_easy.pth")
        # )
        # opponent_ppo_nanyang = PPO(
        #     env=env,
        #     agent=my_agent,
        #     policy_class=FFN,
        #     state_mean=mean,
        #     state_std=std,
        #     **hyperparameters,
        # )
        # opponent_ppo_nanyang.actor.load_state_dict(
        #     torch.load(f"../ppo_actor_nanyang_wcci.pth")
        # )
        # opponent_ppo_d3qn = PPO(
        #     env=env,
        #     agent=my_agent,
        #     policy_class=FFN,
        #     state_mean=None,
        #     state_std=None,
        #     transfer=True,
        #     **hyperparameters2,
        # )
        # opponent_ppo_d3qn.actor.load_state_dict(
        #     torch.load("../ppo_actor_wcci_transfer_0.02clip.pth")
        # )
        # rng1 = np.random.default_rng(i)
        # rng2 = np.random.default_rng(i)
        # opponent_ran = RandomOpponent(
        #     env.observation_space,
        #     env.action_space,
        #     lines_to_attack=attack_lines,
        #     attack_period=attack_period,
        #     attack_duration=attack_duration,
        #     rng=rng1,
        # )
        # opponent_wro = WeightedRandomOpponent(
        #     env.observation_space,
        #     env.action_space,
        #     lines_to_attack=attack_lines,
        #     attack_period=attack_period,
        #     attack_duration=attack_duration,
        #     rng=rng2,
        # )

        trainer_def = TrainAgent(
            my_agent, None, env, env, device, dn_json_path, dn_ffw, ep_infos
        )
        # trainer_ppo_d3qn = TrainAgent(
        #     my_agent,
        #     opponent_ppo_d3qn,
        #     env,
        #     env,
        #     device,
        #     dn_json_path,
        #     dn_ffw,
        #     ep_infos,
        # )
        # trainer_ppo_kaist = TrainAgent(
        #     my_agent,
        #     opponent_ppo_kaist,
        #     env,
        #     env,
        #     device,
        #     dn_json_path,
        #     dn_ffw,
        #     ep_infos,
        # )
        # trainer_ppo_nanyang = TrainAgent(
        #     my_agent,
        #     opponent_ppo_nanyang,
        #     env,
        #     env,
        #     device,
        #     dn_json_path,
        #     dn_ffw,
        #     ep_infos,
        # )
        # trainer_ran = TrainAgent(
        #     my_agent, opponent_ran, env, env, device, dn_json_path, dn_ffw, ep_infos
        # )
        # trainer_wro = TrainAgent(
        #     my_agent, opponent_wro, env, env, device, dn_json_path, dn_ffw, ep_infos
        # )

        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        print("-" * 20 + " No Opponent " + "-" * 20)
        _, tmp_scores, tmp_steps = trainer_def.evaluate(
            test_chronics, MAX_FFW[args.case], output_result_dir + "/no_opp_", mode
        )
        # print("-" * 20 + " PPO D3QN " + "-" * 20)
        # trainer_ppo_d3qn.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/transfer_", mode)
        # print("-" * 20 + " PPO KAIST" + "-" * 20)
        # trainer_ppo_kaist.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/ppo_", mode)
        # print("-" * 20 + " PPO Nanyang" + "-" * 20)
        # trainer_ppo_nanyang.evaluate(
        #     test_chronics, MAX_FFW[args.case], output_result_dir + "/ppo_", mode
        # )
        # print("-" * 20 + " Random Adversary " + "-" * 20)
        # trainer_ran.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/ran_", mode)
        # print("-" * 20 + " Weighted Random Adversary " + "-" * 20)
        # trainer_wro.evaluate(test_chronics, MAX_FFW[args.case], output_result_dir+"/wro_", mode)
