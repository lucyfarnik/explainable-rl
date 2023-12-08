from src import page_grid_world
from src import ppo
from src.maze_env import MazeEnv

if __name__ == "__main__":
    SIZE = 11
    env: MazeEnv = page_grid_world.create_maze(SIZE)
    obs, _ = env.reset()
    # env.render_mode = "human"
    # env.render()
    # env.render_mode = "rgb_array"
    # Square to get grid cells, 3 values per cell, add 1 for direction
    d_obs = SIZE**2 * 3 + 1
    agent = ppo.train_agent(env, d_obs=d_obs, n_act=8)
