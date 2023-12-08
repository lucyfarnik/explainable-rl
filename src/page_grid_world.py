"""The Grid World page of the Streamlit app."""
import gymnasium as gym
from minigrid.core.actions import Actions
from stqdm import stqdm
import streamlit as st
from torch import Tensor, tensor

from src import components
from src.logging import Episode, Step
from src.maze_env import create_maze
from src.parameter import Parameter
from src.ppo import train_agent


POSSIBLE_ACTIONS = [Actions.forward, Actions.left, Actions.right]
MAZE_PARAMETERS = [
    Parameter("Maze Size", default=10.0, min=5.0, max=50.0, unit="corridors across")
]


def page_grid_world():
    """Renders the Grid World page."""
    # Initialise Session State
    for key in ["maze_agent", "maze_test_episodes", "maze_env"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Page Title
    st.title("Grid World")

    # Inputs Section
    components.input_section(MAZE_PARAMETERS)

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

            maze_size = round(st.session_state.maze_size_train_mean)
            # Make environment
            env = create_maze(maze_size)

            # TODO: Get error when calling.
            # Train agent throws an error saying it can't infer the dtype of a dict.
            # The Maze Env passes a dict to the agent while the pole env passes an ndarray of 4 floats.
            # Can the train agent work with a dict?
            # Could override the MazeEnv step and reset methods to return an ndarray instead of a dict.
            # https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/constants.py
            # has the keys for the image in the MazeEnv dict.
            st.session_state.maze_agent = train_agent(env)

            # Store environment in session state
            st.session_state.maze_env = env

    # Testing Section
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
                probs, _ = st.session_state.pole_agent(tensor(obs, dtype=Tensor.float))
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


if __name__ == "__main__":
    page_grid_world()
