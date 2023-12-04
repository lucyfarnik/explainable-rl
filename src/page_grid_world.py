import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper
from random import sample
from stqdm import stqdm
import streamlit as st
from time import sleep
from torch import Tensor, tensor

from src.component_property import properties
from src.logging import Episode, Step
from src.maze_env import MazeEnv
from src.ppo import train_agent


POSSIBLE_ACTIONS = [Actions.forward, Actions.left, Actions.right]


def create_maze(size):
    env = FullyObsWrapper(
        MazeEnv(
            width=size,
            height=size,
            render_mode="rgb_array",
        )
    )
    env.reset()
    return env


# Initialise Session State
if "maze_size" not in st.session_state:
    st.session_state.maze_size = 10

if "maze_env" not in st.session_state:
    st.session_state.maze_env = create_maze(st.session_state.maze_size)

if "maze_agent" not in st.session_state:
    st.session_state.maze_agent = None

# Page Title
st.title("Grid World")


# Inputs Section
st.header("Inputs")
with st.expander("Inputs"):  # Maze controls
    tabs = st.tabs(["Maze Size", "Agent Start Position"])
    with tabs[0]:
        # st.slider(
        #     label="Size",
        #     min_value=5,
        #     value=10,
        #     max_value=50,
        #     step=1,
        #     key="maze_size",
        #     on_change=create_maze(),
        # )
        properties("Size", 5.0, 50.0, 10.0)
    with tabs[1]:
        properties("X Position", 1.0, st.session_state.maze_size * 1.0, 1.0)
        properties("Y Position", 1.0, st.session_state.maze_size * 1.0, 1.0)
    # agent_x_pos, agent_y_pos = agent_position_select()
    # (
    #     position_train_mean,
    #     position_train_std_dev,
    #     position_test_mean,
    #     position_test_std_dev,
    # ) = properties("position", 0.01, 10.00)

# Training Section
st.header("Training")
if st.button(
    label="Train Agent",
    key="train_button",
    help="Train an agent with the current settings.",
):
    with st.spinner(
        "Training agent with your environment parameters..."
    ):  # TODO cache.
        # Delete previous agent and associated data
        for key in ["maze_agent", "maze_test_episodes"]:
            st.session_state[key] = None
        st.session_state.maze_agent = train_agent(
            env=st.session_state.maze_env,
        )


# Testing Section
# st.header("Testing")
# if st.session_state.play:
#     # st.session_state.maze.reset(seed=42)
#     # st.empty is the recommended way to update page content
#     with st.empty():
#         while True:
#             action = sample(POSSIBLE_ACTIONS, 1)[0]
#             st.session_state.maze.step(action=action)
#             img = st.session_state.maze.render()
#             st.image(image=img, caption="Grid World", use_column_width=True)
#             sleep(0.2)
# else:
#     # st.session_state.maze.reset()
#     img = st.session_state.maze.render()
#     st.image(image=img, caption="Grid World", use_column_width=True)
st.header("Testing")
if st.button(
    label="Test Agent",
    key="test_button",
    help="Test the agent with the current settings.",
    disabled=st.session_state.maze_agent is None,
):
    st.session_state.train_test_episodes = []
    for i in stqdm(range(10), desc="Testing agent"):
        env: gym.Env = st.session_state.env
        episode: Episode = Episode(outcome="Success")
        obs, _ = st.session_state.env.reset()
        termination = False
        for step in stqdm(range(400), desc="Step"):
            probs: Tensor
            probs, _ = st.session_state.pole_agent(tensor(obs, dtype=T.float))
            action = probs.argmax(dim=-1)
            action = action.item()
            # st.write(action)
            # obs, reward, termination, _, _ = env.step(agent_action=action)
            obs, reward, termination, _, _ = env.step(action)
            episode.steps.append(
                Step(
                    time_step=step,
                    obs=obs,
                    reward=reward,
                    action=action,
                    render=env.render(),
                )
            )
            if termination:
                obs, _ = st.session_state.env.reset()
                episode.outcome = "Failure"
                break
        st.session_state.pole_test_episodes.append(episode)
