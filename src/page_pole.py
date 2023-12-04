import altair as alt
import gymnasium as gym
from math import degrees
import pandas as pd
from stqdm import stqdm
import streamlit as st
from time import sleep
import torch as T

from src import components
from src.construct_pole_balancing_env import ConstructPoleBalancingEnv
from src.pole_env import PoleEnv
from src.logging import Episode, Step
from src.parameter import Parameter
from src.ppo import train_agent

PARAMETERS = [
    Parameter(name="Gravity", default=9.81, min=0.1, max=20.0, unit="m/s^2"),
    Parameter(
        name="Cart Mass", default=1.0, min=0.5, max=5.0, unit="kg", lib_ref="masscart"
    ),
    Parameter(
        name="Pole Mass", default=0.1, min=0.1, max=1.0, unit="kg", lib_ref="masspole"
    ),
    Parameter(name="Length", default=0.5, min=0.0, max=1.0, unit="m"),
    Parameter(
        name="Force Magnitude",
        default=10.0,
        min=1.0,
        max=20.0,
        unit="N",
        lib_ref="force_mag",
    ),
]


# Initialise Session State
for key in ["pole_agent", "pole_test_episodes", "pole_env"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Construct Environment
# if "pole_env" not in st.session_state:
#     # st.session_state.pole_env = ConstructPoleBalancingEnv(max_iter=400)
#     st.session_state.pole_env = PoleEnv()
#     # st.session_state.pole_env = gym.make("CartPole-v1").unwrapped
#     st.session_state.pole_env.reset()


# Page Title
st.title("Pole Balancing")

# Inputs Section
components.input_section(PARAMETERS)

# Training Section
st.header("Training")
train_button_col, train_result_col = st.columns([1, 4])
with train_button_col:
    if st.button(
        label="Train Agent"
        if st.session_state.pole_agent is None
        else "Re-train Agent",
        key="train_button",
        help="Train an agent with the current settings.",
    ):
        with st.spinner(
            "Training agent with your environment parameters..."
        ):  # TODO cache.
            # Delete previous agent and associated data
            for key in ["pole_agent", "pole_test_episodes"]:
                st.session_state[key] = None
            # Make environment
            env = PoleEnv(
                gravity=(
                    st.session_state.gravity_train_mean,
                    st.session_state.gravity_train_std_dev,
                ),
                mass_cart=(
                    st.session_state.cart_mass_train_mean,
                    st.session_state.cart_mass_train_std_dev,
                ),
                mass_pole=(
                    st.session_state.pole_mass_train_mean,
                    st.session_state.pole_mass_train_std_dev,
                ),
                length=(
                    st.session_state.length_train_mean,
                    st.session_state.length_train_std_dev,
                ),
                force_mag=(
                    st.session_state.force_magnitude_train_mean,
                    st.session_state.force_magnitude_train_std_dev,
                ),
            )
            # Train agent
            st.session_state.pole_agent = train_agent(env)
            # Store environment in session state
            st.session_state.pole_env = env

with train_result_col:
    if st.session_state.pole_agent is not None:
        st.write(st.session_state.pole_agent)


# Testing
st.header("Testing")
if st.button(
    label="Test Agent",
    key="test_button",
    help="Test the agent with the current settings.",
    disabled=st.session_state.pole_agent is None,
):
    # Make environment
    env: gym.Env = PoleEnv(
        gravity=(
            st.session_state.gravity_test_mean,
            st.session_state.gravity_test_std_dev,
        ),
        mass_cart=(
            st.session_state.cart_mass_test_mean,
            st.session_state.cart_mass_test_std_dev,
        ),
        mass_pole=(
            st.session_state.pole_mass_test_mean,
            st.session_state.pole_mass_test_std_dev,
        ),
        length=(
            st.session_state.length_test_mean,
            st.session_state.length_test_std_dev,
        ),
        force_mag=(
            st.session_state.force_magnitude_test_mean,
            st.session_state.force_magnitude_test_std_dev,
        ),
    )
    st.session_state.pole_test_episodes = []
    for i in stqdm(range(10), desc="Testing agent"):
        episode: Episode = Episode(outcome="Success")
        obs, _ = env.reset()
        episode.parameters = {
            param.name_with_unit: getattr(env, param.get_lib_ref())
            for param in PARAMETERS
        }

        termination = False
        for step in stqdm(range(400), desc="Step"):
            probs: T.Tensor
            probs, _ = st.session_state.pole_agent(T.tensor(obs, dtype=T.float))
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
                obs, _ = st.session_state.pole_env.reset()
                episode.outcome = "Failure"
                break
        st.session_state.pole_test_episodes.append(episode)


# TODO move into components >>>
# if "play" not in st.session_state:
#     st.session_state.play = False


# def handle_play_button():
#     st.session_state.play = not st.session_state.play


# Start/Button
# st.button(
#     label=("Stop" if st.session_state.play else "Play"),
#     key="play_button",
#     help="Press to stop and start the animation",
#     on_click=handle_play_button,
# )
# <<<<

# # Display environment
# if st.session_state["play"]:
#     with st.empty():
#         i = 0
#         termination = False
#         obs = None
#         while True:
#             if obs is None:
#                 action = i % 2
#             else:
#                 probs, _ = st.session_state.pole_agent(T.tensor(obs, dtype=T.float))
#                 action = probs.argmax(dim=-1)
#             obs, _, termination, _, _ = st.session_state.pole_env.step(agent_action=action)
#             if termination:
#                 st.session_state.pole_env.reset()
#             img = st.session_state.pole_env.render()
#             st.image(image=img, caption="Pole World", use_column_width=True)
#             sleep(0.1)
#             i += 1
# else:
if st.session_state.pole_test_episodes is not None:
    st.selectbox(
        "Select episode",
        options=st.session_state.pole_test_episodes,
        key="test_episode_select",
    )
    # episode_id: int = st.session_state.test_episode_select
    episode: Episode = st.session_state.test_episode_select

    # Show episode parameters
    param_df = pd.DataFrame.from_records([episode.parameters])
    st.write(param_df)

    step_idx = st.slider(
        "Select step",
        min_value=0,
        max_value=len(episode.steps) - 1,
    )
    step: Step = episode.steps[step_idx]
    render_col, reward_col = st.columns([5, 1])
    with render_col:
        pos, vel, angle, _ = step.obs
        action_text = ":arrow_left:" if step.action == 0 else ":arrow_right:"
        st.write(
            f"Position: {pos:.2f} m | Velocity: {vel:.2f} m/s | Angle: {degrees(angle):.1f}° || Action: {action_text}"
        )
        image = step.render
        st.image(
            image=image,
            caption="Pole World",
            use_column_width=True,
        )
    with reward_col:
        c = (
            alt.Chart(pd.DataFrame({"Reward": [step.reward]}))
            .mark_bar()
            .encode(alt.Y("Reward:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=400)
        )
        st.altair_chart(c, use_container_width=True, theme="streamlit")
