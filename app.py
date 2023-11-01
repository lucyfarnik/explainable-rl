from random import sample
from time import sleep
import streamlit as st
from minigrid.core.actions import Actions
from src.maze_env import MazeEnv


def main():
    possible_actions = [Actions.forward, Actions.left, Actions.right]
    maze = MazeEnv(width=10, height=10, render_mode="rgb_array")
    maze.reset(
        seed=42
    )  # Have to call to run the _gen_grid override and actually generate the grid
    img = maze.render()

    st.title("Grid World")
    # st.empty is the recommended way to update page content
    with st.empty():
        while True:
            action = sample(possible_actions, 1)[0]
            maze.step(action=action)
            img = maze.render()
            st.image(image=img, caption="Grid World", use_column_width=True)
            sleep(0.25)


if __name__ == "__main__":
    main()
