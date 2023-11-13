from random import sample
from time import sleep
import streamlit as st
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper
from src.maze_env import MazeEnv

from src.component_property import agent_position_select, properties, progress


POSSIBLE_ACTIONS = [Actions.forward, Actions.left, Actions.right]


if "maze_size" not in st.session_state:
    st.session_state.maze_size = 10

st.title("Grid World")


def create_maze():
    st.session_state.maze = FullyObsWrapper(
        MazeEnv(
            width=st.session_state.maze_size,
            height=st.session_state.maze_size,
            render_mode="rgb_array",
        )
    )
    st.session_state.maze.reset()
    # Have to call to run the _gen_grid override and actually generate the grid


with st.expander("Inputs"):  # Maze controls
    tabs = st.tabs(["Maze Size", "Agent Start Position"])
    with tabs[0]:
        st.slider(
            label="Size",
            min_value=5,
            value=st.session_state.maze_size,
            max_value=50,
            step=1,
            key="maze_size",
            on_change=create_maze(),
        )
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

    latest_iteration = st.empty()
    bar = st.progress(0)
    "Start a computation..."
    progress()
    "...and we're done"

    # with st.container():
    #     st.button("Up")
    #     st.button("Right")
    #     st.button("Left")
    #     st.button("Down")

# TODO move into components >>>
if "play" not in st.session_state:
    st.session_state.play = False


def handle_play_button():
    st.session_state.play = not st.session_state.play


# Start/Button
st.button(
    ("Stop" if st.session_state.play else "Play"),
    "play_button",
    "Press to stop and start the animation",
    on_click=handle_play_button,
)
# <<<<

if "maze" not in st.session_state:
    create_maze()


if st.session_state.play:
    # st.session_state.maze.reset(seed=42)
    # st.empty is the recommended way to update page content
    with st.empty():
        while True:
            action = sample(POSSIBLE_ACTIONS, 1)[0]
            st.session_state.maze.step(action=action)
            img = st.session_state.maze.render()
            st.image(image=img, caption="Grid World", use_column_width=True)
            sleep(0.2)
else:
    # st.session_state.maze.reset()
    img = st.session_state.maze.render()
    st.image(image=img, caption="Grid World", use_column_width=True)
