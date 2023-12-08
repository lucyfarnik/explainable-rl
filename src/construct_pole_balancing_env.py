"""
This module contains the functions required to construct the environment for 
the RL agent to interact with.
"""
import math
import numpy as np

from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class ConstructPoleBalancingEnv(CartPoleEnv):
    """
    Constructor function to create the environment for the RL task for pole-cart
    problem.

    - Initialise object by calling ConstructPoleBalancingEnv()
    while training:
        - state_transition(agent_action)
        - update_buffer()
        - return_reward()
        - termination_status()

    """

    # assume that this a reasonable time frame for the agent to act and the env
    # need to update
    time_delta = 0.1

    def __init__(
        self,
        cartmass: float | tuple[float, float] = 1.0,
        masspole: float | tuple[float, float] = 0.1,
        gravity: float | tuple[float, float] = 9.81,
        friction: float | tuple[float, float] = 0.0,
        length: float | tuple[float, float] = 0.5,
        cart_x_position: float | tuple[float, float] = 0,
        cart_velocity: float | tuple[float, float] = 0,
        pole_angle: float | tuple[float, float] = 0,
        pole_velocity: float | tuple[float, float] = 0,
        max_iter: int = 100,
        iteration: int = 0,
        maximum_buffer_size: int = 1000,
        force_mag: float = 10.0,
        render_mode: str = "rgb_array",
    ) -> None:
        """
        use the constructor method to define the parameters for the problem.
        vary the parameters to test your agent for different setups

        Parameters
        ----------
        cartmass : float or tuple[float, float]
            Default value is 1 kg
            Can take a single value, or a tuple of mean and std for a normal distribution

        masspole : float or tuple[float, float]
            default value is 0.1 kg
            Can take a single value, or a tuple of mean and std for a normal distribution

        gravity : float or tuple[float, float]
            default value is 9.81 N/kg
            Can take a single value, or a tuple of mean and std for a normal distribution

        friction : float or tuple[float, float]
            default value is frictionless
            Can take a single value, or a tuple of mean and std for a normal distribution

        length : float or tuple[float, float]
            default value is 0.3 meters
            Can take a single value, or a tuple of mean and std for a normal distribution

        cart_x_position : float or tuple[float, float]
            default to the centre of the grid. The centre of the cart is at
            the x coordinate 0
            Can take a single value, or a tuple of mean and std for a normal distribution

        cart_velocity : float or tuple[float, float]
            defaults to stationary
            Can take a single value, or a tuple of mean and std for a normal distribution

        pole_angle : float or tuple[float, float]
            defaults to balanced
            Can take a single value, or a tuple of mean and std for a normal distribution

        pole_velocity : float or tuple[float, float]
            defaults to stationary
            Can take a single value, or a tuple of mean and std for a normal distribution

        max_iter : int
            default to 100 iterations

        iterations: int
            starts as 0 and increments with each time state_transition is called


        Returns
        -------
        None

        """
        super().__init__(render_mode)

        if type(cartmass) is tuple:
            self.cartmass_dist = cartmass
            self.cartmass = np.random.normal(*cartmass)
        else:
            self.cartmass_dist = None
            self.cartmass = cartmass

        if type(masspole) is tuple:
            self.masspole_dist = masspole
            self.masspole = np.random.normal(*masspole)
        else:
            self.masspole_dist = None
            self.masspole = masspole

        if type(gravity) is tuple:
            self.gravity_dist = gravity
            self.gravity = np.random.normal(*gravity)
        else:
            self.gravity_dist = None
            self.gravity = gravity

        if type(friction) is tuple:
            self.friction_dist = friction
            self.friction = np.random.normal(*friction)
        else:
            self.friction_dist = None
            self.friction = friction

        if type(length) is tuple:
            self.length_dist = length
            self.length = np.random.normal(*length)
        else:
            self.length_dist = None
            self.length = length

        if type(cart_x_position) is tuple:
            self.cart_x_position_dist = cart_x_position
            cart_x_position = np.random.normal(*cart_x_position)
        else:
            self.cart_x_position_dist = None

        if type(cart_velocity) is tuple:
            self.cart_velocity_dist = cart_velocity
            cart_velocity = np.random.normal(*cart_velocity)
        else:
            self.cart_velocity_dist = None

        if type(pole_angle) is tuple:
            self.pole_angle_dist = pole_angle
            pole_angle = np.random.normal(*pole_angle)
        else:
            self.pole_angle_dist = None

        if type(pole_velocity) is tuple:
            self.pole_velocity_dist = pole_velocity
            pole_velocity = np.random.normal(*pole_velocity)
        else:
            self.pole_velocity_dist = None

        self.state = [cart_x_position, cart_velocity, pole_angle, pole_velocity]

        self.force_mag = force_mag
        self.max_iter = max_iter
        self.iteration = iteration
        self.maximum_buffer_size = maximum_buffer_size
        self.prev_angle = None

    def return_reward(self, agent_action: float) -> float:
        # return -1 * abs(self.state[2])
        return math.radians(90) - abs(self.state[2])

        # if the last observed angle of the pole is greater is than second to
        # to last observed pole angle i.e. the RL agent caused the pole to
        # be farther away from being balanced. Regardless of the direction
        # if abs(self.state[2]) > abs(self.prev_angle):
        #     # returns a reward negative reward proportional to how farther away
        #     # the pole was diverted from being balanced
        #     return -1 * abs(self.state[2] - self.prev_angle)
        # # if the action made the pole closer to being balanced
        # elif abs(self.state[2]) < abs(self.prev_angle):
        #     # returns a positive reward proportional to how closer it got the
        #     # pole being balanced
        #     return abs(self.state[2] - self.prev_angle)
        # # if the action didn't change the angle
        # else:
        #     return 0.0

    def state_transition(self, agent_action: float):
        """
        positive agent action is a force to the right.

        Parameters
        ----------
        agent_action : int
            direction to apply force by the agent on the cart .

        Returns
        -------
        dict
            dictionary with the current state observation for the env.

        """
        # update the cart's state as per the agent's action (Force on cart)

        # determine if the force should be + or - according to direction
        # received from the agent
        if agent_action == 1:
            force = self.force_mag
        else:
            force = -self.force_mag

        # retrieve the current values of the components of the state
        cart_position = self.state[0]
        cart_velocity = self.state[1]
        pole_angle = self.state[2]
        self.prev_angle = pole_angle
        pole_velocity = self.state[3]

        #  update force according to friction
        force = force - self.friction

        total_mass = self.masspole + self.cartmass
        pole_mass_length = self.masspole * self.length / 2

        # intermediate variable to facilitate computation of the acc
        temp = (
            force
            + pole_mass_length * pole_velocity**2 * np.sin(pole_angle)
            - pole_velocity * np.cos(pole_angle)
        ) / total_mass

        # update the pole acc and cart acc
        pole_acceleration = (
            self.gravity * np.sin(pole_angle) - np.cos(pole_angle) * temp
        ) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.cos(pole_angle) ** 2 / total_mass)
        )
        cart_acceleration = (
            temp
            - pole_mass_length * pole_acceleration * np.cos(pole_angle) / total_mass
        )

        # Update state variables
        cart_position = cart_position + cart_velocity
        cart_velocity = cart_velocity + cart_acceleration
        pole_angle = pole_angle + pole_velocity
        pole_velocity = pole_velocity + pole_acceleration

        # increment the current iteration
        self.iteration += 1
        self.state = [cart_position, cart_velocity, pole_angle, pole_velocity]

        return self.state

    def termination_status(self) -> bool:
        """
        terminate if the pole is at 90 degree angle with the y-axis from either
        side or if the max iteration is reached

        Returns
        -------
        bool
            1 means terminate and 0 is do not terminate.

        """
        if (
            self.state[2] >= math.radians(90)
            or self.state[2] <= math.radians(-90)
            or self.iteration >= self.max_iter
        ):
            return 1
        else:
            return 0

    def step(self, agent_action):
        """
        Call to construct the environment and observe states without
        having to call each method on its own in your script
        """
        # first get the new env observation
        observation = self.state_transition(agent_action)
        # compute the reward
        reward = self.return_reward(agent_action)
        termination = self.termination_status()
        return observation, reward, termination, False, dict()

    def reset(self, *args, **kwargs):
        self.iteration = 0
        super().reset(*args, **kwargs)
        if self.cartmass_dist is not None:
            self.cartmass = np.random.normal(*self.cartmass_dist)
        if self.masspole_dist is not None:
            self.masspole = np.random.normal(*self.masspole_dist)
        if self.gravity_dist is not None:
            self.gravity = np.random.normal(*self.gravity_dist)
        if self.friction_dist is not None:
            self.friction = np.random.normal(*self.friction_dist)
        if self.length_dist is not None:
            self.length = np.random.normal(*self.length_dist)
        if self.cart_x_position_dist is not None:
            self.state[0] = np.random.normal(*self.cart_x_position_dist)
        if self.cart_velocity_dist is not None:
            self.state[1] = np.random.normal(*self.cart_velocity_dist)
        if self.pole_angle_dist is not None:
            self.state[2] = np.random.normal(*self.pole_angle_dist)
        if self.pole_velocity_dist is not None:
            self.state[3] = np.random.normal(*self.pole_velocity_dist)

        return np.array(self.state, dtype=np.float32), {}
