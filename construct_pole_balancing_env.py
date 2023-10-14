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

class ConstructPoleBalancingEnv():
    
    def __init__(
            self,
            cart_mass: float = 1.0, 
            pole_mass: float = 0.3, 
            gravity: float = 9.81, 
            friction: float = 0.0, 
            pole_length: float = 0.5
            )->None:
        """
        use the constructor method to define the parameters for the problem. 
        vary the parameters to test your agent for different setups

        Parameters
        ----------
        cart_mass : float
            Default value is 1 kg

        pole_mass : float
            default value is 0.3 kg
            
        gravity : float
            default value is 9.81 N/kg

        friction : float
            default value is frictionless 

        pole_length : float
            default value is 0.3 meters


        Returns
        -------
        None

        """
        self.cart_mass=cart_mass
        self.pole_mass=pole_mass
        self.friction=friction
        self.gravity=gravity
        self.pole_length=pole_length
        