"""
Creates a maze environment. The environment is based on the MiniGrid package.
The mazelib package is used to 

Todo:
    * Make the agent start position and goal configurable. 
"""
from __future__ import annotations

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Goal  # , Door, Key #TODO
# from minigrid.core.constants import COLOR_NAMES # Used for doors and keys

from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms  # TODO add options


class MazeEnv(MiniGridEnv):
    """
    Maze Environment based on the MiniGrid package.

    Attributes:
        width (int): Number of corridors across the maze (excluding walls)
        heigh (int): Number of corridors vertically (excluding walls)
        agent_start_pos (tuple(int, int)): X and Y coordinates the agent starts from.
            Coordinates are 0 indexed but the 0th column and row will be in the outer wall so (1, 1)
            is the top left corner.
        agent_start_dir (int): Direction the agent is facing at start. 0 -> north, 1 -> east, 2 -> south, 3 -> west.
        max_steps (int): Number of steps the agent can take before the simulation ends.
        render_mode (str): Inherited. Either "human" or "rgb_array". If "human", the inherited function render()
            opens a window to display the state. If "rgb_array" returns an n x m x 3 numpy array of rgb pixel values.

    Example:
        >>> maze = MazeEnv(width=10, height=10, render_mode="rgb_array")
        >>> maze.reset() # Have to call to run the _gen_grid override and actually generate the grid
        >>> img = maze.render()

    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        agent_start_pos: tuple(int, int) = (1, 1),
        agent_start_dir: int = 0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * width * height

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        """Simply creates a string that describes the agent's mission in the MiniGrid Environment.

        Returns:
            str: The mission as a human-readable string.
        """
        return "Solve the Maze"

    def _gen_grid(self, width, height):
        """
        Overrides an empty function on the MiniGrid class.
            The intended way to define the grid construction function.
            Must call reset() after initializing to run this function.

        Args:
            width (int): Width of the maze in corridors (excluding walls)
            height (int): Height of the maze in corridors (excluding walls)
        """
        # Generate Maze using the mazelib package
        m = Maze()
        m.generator = DungeonRooms(h0=height, w0=width)  # TODO add options
        m.generate()

        # Mazelib counts height and width as the corridors, excluding the walls
        # As such it adds cells to allow it to fill in cells as walls.
        # e.g. h = h * 2 + 1
        # Create an empty grid
        self.width = m.grid.shape[0]  # Update the grid size based on the mazelib size
        self.height = m.grid.shape[1]  # Update the grid size based on the mazelib size
        self.grid = Grid(self.width, self.height)

        # Mazelib returns an h x w numpy array with 1s representing walls
        # MiniGrid stores the grid as  flat array
        # MiniGrid does provide a set function that can set a specific cell's value, but this would require looping.
        # It doesn't protect setting the grid directly so I'm using a list comp.
        self.grid.grid = [Wall() if el == 1 else None for el in m.grid.flatten()]

        # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square
        self.put_obj(
            Goal(), self.width - 2, self.height - 2
        )  # in the bottom-right corner for now

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
