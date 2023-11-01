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
    time_delta=0.1
    
    def __init__(
            self,
            cartmass: float = 1.0, 
            masspole: float = 0.0, 
            gravity: float = 9.81, 
            friction: float = 0.0, 
            length: float = 0.5, 
            cart_x_position: float = 0, 
            cart_velocity: float = 0, 
            pole_angle: int = 0, 
            pole_velocity: float = 0,
            max_iter: int = 100,
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
        
    def return_reward(
            self, 
            agent_action: float
            )->float:
        # if the last observed angle of the pole is greater is than second to
        # to last observed pole angle i.e. the RL agent caused the pole to 
        # be farther away from being balanced. Regardless of the direction
        if abs(self.state[2])>abs(self.prev_angle):
            # returns a reward negative reward proportional to how farther away
            # the pole was diverted from being balanced
            return -1*(abs(self.state[2]-self.prev_angle)/45)
        # if the action made the pole closer to being balanced
        elif abs(self.state[2])<abs(self.prev_angle):
            # returns a positive reward proportional to how closer it got the
            # pole being balanced
            return (abs(self.state[2]-self.prev_angle)/45)
        # if the action didn't change the angle
        else:
            return 0.0
        
        
    def state_transition(
            self,
            agent_action: float
            ):
        """
        positive agent action is a force to the right.

        Parameters
        ----------
        agent_action : float
            force applied by the agent on the cart .

        Returns
        -------
        dict
            dictionary with the current state observation for the env.

        """
        # update the cart's state as per the agent's action (Force on cart)
        
        cart_position=self.state[0]
        cart_velocity=self.state[1]
        pole_angle=self.state[2]
        self.prev_angle=pole_angle
        pole_velocity=self.state[3]
        friction_force=self.friction*cart_velocity
        cart_acceleration=(agent_action-friction_force)/self.cartmass
        cart_position=cart_position+(cart_velocity*self.time_delta)
        cart_velocity=cart_velocity+(cart_acceleration*self.time_delta)
        
        # update the pole's state
        pole_angular_acceleration=(self.gravity/self.length)*math.sin(pole_angle)
        pole_angle=pole_angle+(pole_angular_acceleration*self.time_delta)
        pole_velocity=pole_velocity+(pole_angular_acceleration*self.time_delta)
        
        # increment the current iteration 
        self.iteration+=1
        self.state=[cart_position, cart_velocity, pole_angle, pole_velocity]
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
        if (
                self.state[2]>=90 or
                self.state[2]<=-90 or
                self.iteration>=self.max_iter
                ):
            print("Problem cannot be solved anymore or maximum iteration reached")
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
        return [observation, reward, termination]
        
        
        
        
