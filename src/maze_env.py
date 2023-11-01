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
    Creates a Maze Environment
    """

    def __init__(
        self,
        width=10,
        height=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
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
        return "Solve the Maze"

    # Overrides an empty function on the MiniGrid class
    # The intended way to define the grid construction function.
    # Must call reset() after constructing to run this function.
    def _gen_grid(self, width, height):
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
