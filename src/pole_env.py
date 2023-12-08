"""A wrapper for the CartPole environment from gymnasium that allows the parameters to be sampled from distributions."""
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np


class PoleEnv(CartPoleEnv):
    """A wrapper for the CartPole environment from gymnasium that allows the parameters to be sampled from distributions.

    Attributes:
        gravity (tuple[float, float]): Gravity (m/s^2).
        mass_cart (tuple[float, float]): Mass of the cart (kg).
        mass_pole (tuple[float, float]): mass of the pole (kg).
        length (tuple[float, float]): The length of the pole (m).
        force_mag (tuple[float, float]): The magnitude of the force the agent can apply to the cart (either left or right) each time step (N).

        For each attribute, you can specify a float or a tuple of floats. A float sets the value of the attribute to that value.
        If providing a tuple, specify a (mean, std_dev) pair. The environment will sample from that normal distribution each time it is reset.
    """

    def __init__(
        self,
        gravity: float | tuple[float, float] = 9.8,
        mass_cart: float | tuple[float, float] = 1.0,
        mass_pole: float | tuple[float, float] = 0.1,
        length: float | tuple[float, float] = 1,
        force_mag: float | tuple[float, float] = 10.0,
    ):
        super().__init__()
        if type(gravity) is tuple:
            self.gravity_dist = gravity
            self.gravity = np.random.normal(*gravity)
        else:
            self.gravity_dist = None
            self.gravity = gravity

        if type(mass_cart) is tuple:
            self.masscart_dist = mass_cart
            self.masscart = np.random.normal(*mass_cart)
        else:
            self.masscart_dist = None
            self.masscart = mass_cart

        if type(mass_pole) is tuple:
            self.masspole_dist = mass_cart
            self.masspole = np.random.normal(*mass_pole)
        else:
            self.masspole_dist = None
            self.masspole = mass_pole

        self.total_mass = self.masspole + self.masscart

        if type(length) is tuple:
            self.length_dist = mass_cart
            self.length = np.random.normal(*length) / 2
        else:
            self.length_dist = None
            self.length = length / 2

        self.polemass_length = self.masspole * self.length

        if type(force_mag) is tuple:
            self.force_mag_dist = force_mag
            self.force_mag = np.random.normal(*force_mag)
        else:
            self.force_mag_dist = None
            self.force_mag = force_mag

        self.render_mode = "rgb_array"

    def reset(self):
        super_return = super().reset()
        if self.gravity_dist is not None:
            self.gravity = np.random.normal(*self.gravity_dist)
        if self.masscart_dist is not None:
            self.masscart = np.random.normal(*self.masscart_dist)
        self.total_mass = self.masspole + self.masscart
        if self.masspole_dist is not None:
            self.masspole = np.random.normal(*self.masspole_dist)
        if self.length_dist is not None:
            self.length = np.random.normal(*self.length_dist) / 2
        self.polemass_length = self.masspole * self.length

        return super_return

    def step(self, action):
        state, reward, terminated, _, _ = super().step(action)

        reward = ((5 - abs(state[0])) / 5 + (0.5 - abs(state[2])) / 0.5) / 2

        return state, reward, terminated, False, {}
