import streamlit as st
import torch as T
from time import sleep

from src.component_property import properties
from src.construct_pole_balancing_env import ConstructPoleBalancingEnv
from src.ppo import train_agent

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
    st.session_state.env = ConstructPoleBalancingEnv()
    st.session_state.env.reset()


if "agent" not in st.session_state:
    st.session_state.agent = None


# Training
if st.button(
    label="Train Agent",
    key="train_button",
    help="Train an agent with the current settings.",
):
    with st.spinner("Training agent..."):
        st.session_state.agent = None
        st.session_state.agent = train_agent(
            env=st.session_state.env,
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

# Display environment
if st.session_state["play"]:
    with st.empty():
        i = 0
        termination = False
        obs = None
        while True:
            if obs is None:
                action = i % 2
            else:
                probs, _ = st.session_state.agent(T.tensor(obs, dtype=T.float))
                action = probs.argmax(dim=-1)
            obs, _, termination, _, _ = st.session_state.env.step(agent_action=action)
            if termination:
                st.session_state.env.reset()
            img = st.session_state.env.render()
            st.image(image=img, caption="Pole World", use_column_width=True)
            sleep(0.1)
            i += 1
else:
    st.image(
        image=st.session_state.env.render(), caption="Pole World", use_column_width=True
    )  # Finish with cart on screen
