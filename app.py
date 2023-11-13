from random import sample
from time import sleep
import streamlit as st
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper
from src.maze_env import MazeEnv

POSSIBLE_ACTIONS = [Actions.forward, Actions.left, Actions.right]


def main():
    st.title("Grid World")

    with st.container():  # Maze controls
        input_width = st.slider(
            label="Size", min_value=5, value=10, max_value=50, step=1
        )

    maze = FullyObsWrapper(
        MazeEnv(width=input_width, height=input_width, render_mode="rgb_array")
    )
    obs, _ = maze.reset(
        seed=42
    )  # Have to call to run the _gen_grid override and actually generate the grid
    img = maze.render()

    # st.empty is the recommended way to update page content
    with st.empty():
        while True:
            action = sample(POSSIBLE_ACTIONS, 1)[0]
            maze.step(action=action)
            img = maze.render()
            st.image(image=img, caption="Grid World", use_column_width=True)
            sleep(0.2)


if __name__ == "__main__":
    main()
