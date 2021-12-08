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

from simple_opponents.random_opponent import RandomOpponent, WeightedRandomOpponent
from ppo.ppo import PPO
from ppo.nnpytorch import FFN

import matplotlib
from matplotlib import pyplot as plt
from grid2op.PlotGrid import PlotMatplot

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


def save_frames_as_gif(frames, path="./", filename="gym_animation.gif"):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])

    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=60)


def evaluate(env, agent, opponent, n_episodes, max_steps):
    print("*" * 80)
    plot_helper = PlotMatplot(env.observation_space)
    reward_arr, n_survive_steps_arr = [], []
    for i_episode in range(1, n_episodes + 1):
        step = 0
        done = False
        obs = env.reset()
        agent.reset(obs)
        opponent.reset(obs)
        opponent._next_attack_time = 2
        total_reward = []
        frames = []

        while step < max_steps and not done:
            env.render()
            print(obs.line_status)
            # agent act
            step += 1
            a = agent.act(obs, None, None)
            obs, reward, done, info = env.step(a)
            total_reward.append(reward)
            if done:
                break

            # opponent attack
            if opponent:
                if opponent.remaining_time >= 0:
                    obs.time_before_cooldown_line[
                        opponent.attack_line
                    ] = opponent.remaining_time
                    opponent.remaining_time -= 1
                else:
                    attack = opponent.act(obs, None, None)
                    if attack is not None:
                        step += 1
                        print("Attack")
                        print(attack)
                        obs, opp_reward, done, info = env.step(attack)
                        obs.time_before_cooldown_line[
                            opponent.attack_line
                        ] = opponent.remaining_time
                        total_reward.append(-1 * opp_reward)
            env.render()
            print(obs.line_status)
            if done:
                break

        reward_arr.append(total_reward)
        n_survive_steps_arr.append(step)

    return reward_arr, n_survive_steps_arr, frames


def main():
    matplotlib.use("Agg")
    args = cli()

    # settings
    model_name = args.name  # + "_" + args.opp + "_adv_" + str(args.seed)
    print("model_name: ", model_name)

    OUTPUT_DIR = "./result"
    DATA_DIR = "./data"
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
    env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 5
    env.parameters.NB_TIMESTEP_RECONNECTION = 12
    env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    env.parameters.HARD_OVERFLOW_THRESHOLD = 2.0
    print(env.parameters.__dict__)

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

    REDUCED_LINES = [
        "0_4_2",
        "10_11_11",
        "11_12_13",
        "12_13_14",
        "12_16_20",
        "13_14_15",
        "13_15_16",
        "14_16_17",
        "14_35_53",
        "15_16_21",
        "16_17_22",
        "16_18_23",
        "16_21_27",
        "16_21_28",
        "16_33_48",
        "16_33_49",
        "16_35_54",
        "17_24_33",
        "18_19_24",
        "18_25_35",
        "19_20_25",
        "1_10_12",
        "1_3_3",
        "1_4_4",
        "20_21_26",
        "21_22_29",
        "21_23_30",
        "21_26_36",
        "22_23_31",
        "22_26_39",
    ]

    LINES = [
        "0_4_2",
        "10_11_11",
        "11_12_13",
        "12_13_14",
        "12_16_20",
        "13_14_15",
        "13_15_16",
        "14_16_17",
        "14_35_53",
        "15_16_21",
        "16_17_22",
        "16_18_23",
        "16_21_27",
        "16_21_28",
        "16_33_48",
        "16_33_49",
        "16_35_54",
        "17_24_33",
        "18_19_24",
        "18_25_35",
        "19_20_25",
        "1_10_12",
        "1_3_3",
        "1_4_4",
        "20_21_26",
        "21_22_29",
        "21_23_30",
        "21_26_36",
        "22_23_31",
        "22_26_39",
        "23_24_32",
        "23_25_34",
        "23_26_37",
        "23_26_38",
        "26_27_40",
        "26_28_41",
        "26_30_56",
        "27_28_42",
        "27_29_43",
        "28_29_44",
        "28_31_57",
        "29_33_50",
        "29_34_51",
        "2_3_0",
        "2_4_1",
        "30_31_45",
        "31_32_47",
        "32_33_58",
        "33_34_52",
        "4_5_55",
        "4_6_5",
        "4_7_6",
        "5_32_46",
        "6_7_7",
        "7_8_8",
        "7_9_9",
        "8_9_10",
        "9_16_18",
        "9_16_19",
    ]

    SAND_LINES = ["1_2_2", "1_4_4", "2_3_5", "6_8_19", "8_13_11", "9_10_12"]
    EASY_PARL = [
        "34_35_110",
        "43_44_125",
        "52_53_139",
        "54_58_154",
        "37_36_179",
        "63_64_163",
        "48_65_165",
        "55_57_148",
        "45_46_127",
        "38_39_119",
    ]

    # 342 for kaist-sand, 420 for default-sand
    attack_period = 50
    attack_duration = 10
    attack_lines = SAND_LINES

    hyperparameters = {
        "lines_attacked": attack_lines,
        "attack_duration": attack_duration,
        "attack_period": attack_period,
        "danger": 0.9,
        "state_dim": 342,
    }

    hyperparameters2 = {
        "lines_attacked": attack_lines,
        "attack_duration": attack_duration,
        "attack_period": attack_period,
        "danger": 0.9,
        "state_dim": 420,
    }

    data_dir = os.path.join("../kaist_agent/sand_data")
    with open(os.path.join(data_dir, "param.json"), "r", encoding="utf-8") as f:
        param = json.load(f)
    # print(param)
    mean = torch.load(os.path.join(data_dir, "mean.pt"), map_location="cpu").cpu()
    std = torch.load(os.path.join(data_dir, "std.pt"), map_location="cpu").cpu()

    # opponent_ppo_kaist = PPO(env=env, agent=my_agent, policy_class=FFN, state_mean=mean, state_std=std, **hyperparameters)
    # opponent_ppo_kaist.actor.load_state_dict(torch.load('../ppo_actor_sandbox.pth'))
    rng1 = np.random.default_rng(0)
    opponent_ran = RandomOpponent(
        env.observation_space,
        env.action_space,
        lines_to_attack=attack_lines,
        attack_period=attack_period,
        attack_duration=attack_duration,
        rng=rng1,
    )
    print(evaluate(env, my_agent, opponent_ran, 1, 100))


if __name__ == "__main__":
    main()
