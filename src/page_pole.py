from math import degrees
from numpy import ndarray
import pandas as pd
import streamlit as st
import torch as T
import altair as alt
from time import sleep
from stqdm import stqdm
from dataclasses import dataclass, field
import gymnasium as gym

from src.component_property import properties
from src.construct_pole_balancing_env import ConstructPoleBalancingEnv
from src.ppo import train_agent


@dataclass
class Step:
    time_step: int
    obs: T.Tensor
    reward: float
    action: int
    render: ndarray

    def __str__(self) -> str:
        return f"t={self.time_step}"


@dataclass
class Episode:
    steps: list[Step] = field(default_factory=list)
    outcome: str = field(default="")

    def length(self) -> int:
        return len(self.steps)

    def __str__(self) -> str:
        return f"Episode: {self.outcome} - {self.length()} steps"


st.title("Pole Balancing")

with st.expander("Inputs"):
    tabs = st.tabs(["Cart Mass", "Pole Mass", "Gravity", "Friction", "Length", "Angle"])
    with tabs[0]:
        cart_mass = properties("cart mass", min=0.5, max=5.0, default=1.0)
    with tabs[1]:
        pole_mass = properties("pole mass", min=0.1, max=1.0, default=0.1)
    with tabs[2]:
        gravity = properties("gravity", min=0.1, max=20.0, default=9.81)
    with tabs[3]:
        friction = properties("friction", min=0.0, max=1.0, default=0.0)
    with tabs[4]:
        length = properties("length", min=0.0, max=1.0, default=0.5)
    with tabs[5]:
        pole_start_angle = properties("angle", min=0.0, max=45.0, default=0.0)


# Construct Environment
if "env" not in st.session_state:
    # st.session_state.env = ConstructPoleBalancingEnv(max_iter=400, cartmass=5.0)
    st.session_state.env = gym.make("CartPole-v1").unwrapped
    st.session_state.env.reset()


if "agent" not in st.session_state:
    st.session_state.agent = None


# Training
if st.button(
    label="Train Agent",
    key="train_button",
    help="Train an agent with the current settings.",
):
    with st.spinner("Training agent..."):  # TODO cache.
        # Delete previous agent and associated data
        for key in ["agent", "test_episodes"]:
            if key in st.session_state:
                del st.session_state[key]

        st.session_state.agent = train_agent(
            env=st.session_state.env,
        )

# st.write(st.session_state)

# Testing
if st.button(
    label="Test Agent",
    key="test_button",
    help="Test the agent with the current settings.",
    disabled=st.session_state.agent is None,
):
    st.session_state.test_episodes = []
    for i in stqdm(range(10), desc="Testing agent"):
        env: gym.Env = st.session_state.env
        env.render_mode = "rgb_array"
        episode: Episode = Episode(outcome="Success")
        obs, _ = st.session_state.env.reset()
        termination = False
        for step in stqdm(range(400), desc="Step"):
            probs: T.Tensor
            probs, _ = st.session_state.agent(T.tensor(obs, dtype=T.float))
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
        st.session_state.test_episodes.append(episode)

if "test_episodes" in st.session_state:
    st.selectbox(
        "Select episode",
        options=st.session_state.test_episodes,
        key="test_episode_select",
    )

# TODO move into components >>>
if "play" not in st.session_state:
    st.session_state.play = False


def handle_play_button():
    st.session_state.play = not st.session_state.play


# Start/Button
st.button(
    label=("Stop" if st.session_state.play else "Play"),
    key="play_button",
    help="Press to stop and start the animation",
    on_click=handle_play_button,
)
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
#                 probs, _ = st.session_state.agent(T.tensor(obs, dtype=T.float))
#                 action = probs.argmax(dim=-1)
#             obs, _, termination, _, _ = st.session_state.env.step(agent_action=action)
#             if termination:
#                 st.session_state.env.reset()
#             img = st.session_state.env.render()
#             st.image(image=img, caption="Pole World", use_column_width=True)
#             sleep(0.1)
#             i += 1
# else:
if "test_episodes" in st.session_state:
    # episode_id: int = st.session_state.test_episode_select
    episode: Episode = st.session_state.test_episode_select
    if not st.session_state.play:
        step_idx = st.slider(
            "Select step",
            min_value=0,
            max_value=len(episode.steps) - 1,
        )
    step: Step = episode.steps[step_idx]
    render_col, reward_col = st.columns([5, 1])
    with render_col:
        pos, vel, angle, _ = step.obs
        st.write(
            f"Position: {pos:.2f} m | Velocity: {vel:.2f} m/s | Angle: {degrees(angle):.1f}°"
        )
        image = step.render
        st.image(
            image=image,
            caption="Pole World",
            use_column_width=True,
        )
        action_text = ":arrow_left:" if step.action == 0 else ":arrow_right:"
        st.write("Action:", action_text)
    with reward_col:
        c = (
            alt.Chart(pd.DataFrame({"Reward": [step.reward]}))
            .mark_bar()
            .encode(alt.Y("Reward:Q", scale=alt.Scale(domain=[0, 1])))
            .properties(height=400)
        )
        st.altair_chart(c, use_container_width=True, theme="streamlit")
