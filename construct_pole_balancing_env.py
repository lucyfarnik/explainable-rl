"""
This module contains the functions required to construct the environment for 
the RL agent to interact with. The following methods are included:
    - env_params_setup (cart mass, pole mass, gravity, friction of the surface, pole length))
    - state_space_define
    - state_transition_function
    - reward_function
    - episode_loop
    - env_reset

@author: hn23952
"""
import math 
import numpy as np

from gymnasium.envs.classic_control.cartpole import CartPoleEnv

class ConstructPoleBalancingEnv(CartPoleEnv):
    """
    Construtor function to create the environment for the RL task for pole-cart
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
    time_delta=0.02
    
    def __init__(
            self,
            cartmass: float = 1.0, 
            masspole: float = 0.1, 
            gravity: float = 9.81, 
            friction: float = 0.0, 
            length: float = 0.5, 
            cart_x_position: float = 0, 
            cart_velocity: float = 0, 
            pole_angle: int = 0.2, 
            pole_velocity: float = 0,
            max_iter: int = 1000,
            iteration: int =0,
            maximum_buffer_size: int = 1000,
            force_mag: float = 10.0, 
            render_mode: str = "human"

            )->None:
        """
        use the constructor method to define the parameters for the problem. 
        vary the parameters to test your agent for different setups

        Parameters
        ----------
        cartmass : float
            Default value is 1 kg

        masspole : float
            
        gravity : float
            default value is 9.81 N/kg

        friction : float
            default value is frictionless 

        length : float
            default value is 0.3 meters
        
        cart_x_postion : float
            default to the centre of the grid. The centre of the cart is at 
            the x coordinate 0
            
        cart_velocity : float
            defaults to stationary
            
        pole_angle : int
            defaults to balanced
        
        pole_velocity : float
            defaults to stationary
            
        max_iter : int
            default to 100 iterations
            
        iterations: int
            starts as 0 and increments with each time state_transition is called


        Returns
        -------
        None

        """
        super().__init__(render_mode)
        self.cartmass=cartmass
        self.masspole=masspole
        self.friction=friction
        self.gravity=gravity
        self.force_mag=force_mag
        self.length=length
        self.state=[cart_x_position, cart_velocity, pole_angle, pole_velocity]
        self.max_iter=max_iter
        self.iteration=iteration
        self.maximum_buffer_size=maximum_buffer_size
        self.prev_angle=None
        self.pole_mass_length=self.masspole*self.length
        
    def return_reward(
            self, 
            agent_action: float
            )->float:

        # return -1*abs(self.state[2])
        
        # Define a threshold for pole angle and position
        angle_threshold = 20  # in degrees
        # position_threshold = 2.4  # in units
    
        # Check if the pole has fallen or if the cart has gone out of bounds
        if abs(self.state[2]) > math.radians(angle_threshold):
            return 0  # Penalize for failure
    
        # Otherwise, provide a small positive reward for each time step
        return 1

        
    def state_transition(
            self,
            agent_action: float
            ):
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
            force=self.force_mag
        else:
            force=-self.force_mag
            
        # retrieve the current values of the components of the state        
        cart_position, cart_velocity, pole_angle, pole_velocity = self.state
        self.prev_angle=pole_angle
        # force = self.force_mag if action == 1 else -self.force_mag
        cospole_angle = math.cos(pole_angle)
        sinpole_angle = math.sin(pole_angle)
        
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * pole_velocity**2 * sinpole_angle
        ) / self.total_mass
        pole_angle_acc = (self.gravity * sinpole_angle - cospole_angle * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cospole_angle**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * pole_angle_acc * cospole_angle / self.total_mass
        
        #update the state variables
        cart_position = cart_position + self.tau * cart_velocity
        cart_velocity = cart_velocity + self.tau * xacc
        pole_angle = pole_angle + self.tau * pole_velocity
        pole_velocity = pole_velocity + self.tau * pole_angle_acc
        
        
        self.state = [cart_position, cart_velocity, pole_angle, pole_velocity]
        
        # increment the current iteration 
        self.iteration+=1
        # self.state=[cart_position, cart_velocity, pole_angle, pole_velocity]
        
        return self.state
    
    def termination_status(self)->bool:
        """
        terminate if the pole is at 90 degree angle with the y-axis from either
        side or if the max iteration is reached

        Returns
        -------
        bool
            1 means terminate and 0 is do not terminate.

        """
        if abs(self.state[2])>=math.radians(90): 
                # self.iteration>=self.max_iter
            print(self.state[2])
            print('terminating')
            return 1
        else:
            
            return 0
        
    def step(self, agent_action):
        """
            Call to construct the environment and observe states without
            having to call each method on its own in your script
        """
        # first get the new env observation
        observation=self.state_transition(agent_action)
        # compute the reward
        reward=self.return_reward(agent_action)
        termination=self.termination_status()
        return observation, reward, termination, False, dict()
        
    def reset(self, *args, **kwargs):
        self.iteration=0
        return super().reset(*args, **kwargs)
        
        
        
