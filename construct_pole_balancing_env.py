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

class ConstructPoleBalancingEnv():
    
    # assume that this a reasonable time frame for the agent to act and the env
    # need to update
    time_delta=0.1
    
    def __init__(
            self,
            cart_mass: float = 1.0, 
            cart_length: float = 5,
            cart_height: float = 5,
            pole_mass: float = 0.0, 
            gravity: float = 9.81, 
            friction: float = 0.0, 
            pole_length: float = 0.5, 
            cart_x_position: float = 0, 
            cart_velocity: float = 0, 
            pole_angle: int = 0, 
            pole_velocity: float = 0, 
            
            )->None:
        """
        use the constructor method to define the parameters for the problem. 
        vary the parameters to test your agent for different setups

        Parameters
        ----------
        cart_mass : float
            Default value is 1 kg

        pole_mass : float
        
        cart_length : float
        
        cart_height : float
            
        gravity : float
            default value is 9.81 N/kg

        friction : float
            default value is frictionless 

        pole_length : float
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


        Returns
        -------
        None

        """
        self.cart_mass=cart_mass
        self.pole_mass=pole_mass
        self.cart_length=cart_length
        self.cart_height=cart_height
        self.friction=friction
        self.gravity=gravity
        self.pole_length=pole_length
        self.cart_x_position=cart_x_position
        self.cart_velocity=cart_velocity
        self.pole_angle=pole_angle
        self.pole_velocity=pole_velocity
        self.__cart_position_buffer=[cart_x_position]
        self.__cart_velocity_buffer=[cart_velocity]
        self.__pole_angle_buffer=[pole_angle]
        self. __pole_velocity_buffer=[pole_velocity]
        

    def update_buffer(
            self,
            cart_position: float, 
            cart_velocity: float, 
            pole_angle: int, 
            pole_velocity: float
            )->[list, list, list, list]:
        """
        A method to update the buffers with new space state of the enviroment
        The buffers could be used to use the observation at the previous 
        time step to satisify the MDP formulation or for experience replay 
        in training the RL agent 

        Parameters
        ----------
        cart_position : float
            DESCRIPTION.
        cart_velocity : float
            DESCRIPTION.
        pole_angle : int
            DESCRIPTION.
        pole_velocity : float
            DESCRIPTION.

        Returns
        -------
        [list, list, list, list]
            [cart_position_buffer, cart_velocity_buffer, pole_angle_buffer, pole_velocity_buffer].

        """
        # update private variables using this method. The variables are made
        # private to ensure they are not corrupted outside the class by mistake 
        self.__cart_position_buffer.append(cart_position)
        self.__cart_velocity_buffer.append(cart_velocity)
        self.__pole_angle_buffer.append(pole_angle)
        self.__pole_velocity_buffer.append(pole_velocity)
        
        
        return [self.__cart_position_buffer,
                self.__cart_velocity_buffer, 
                self.__pole_angle_buffer,
                self.__pole_velocity_buffer]
    
    def view_buffer(self)-> [list, list, list, list]:
        
        print("the cart position buffer = ", self.__cart_position_buffer)
        print("The cart velocity buffer = ", self.__cart_velocity_buffer)
        print("The pole angle buffer = ", self.__pole_angle_buffer)
        print("The pole velocity buffer = ", self.__pole_velocity_buffer)
        
        return [self.__cart_position_buffer, self.__cart_velocity_buffer, 
                self.__pole_angle_buffer, self.__pole_velocity_buffer]
    
    def return_reward(
            self
            ):
        
        # if the last observed angle of the pole is greater is than second to
        # to last observed pole angle i.e. the RL agent caused the pole to 
        # be farther away from being balanced. Regardless of the direction
        if abs(self.pole_angle_buffer[-1])>abs(self.pole_angle_buffer[-2]):
            # returns a reward negative reward proportional to how farther away
            # the pole was diverted from being balanced
            return -1*(abs(self.pole_angle_buffer[-1])/45)
        # if the action made the pole closer to being balanced
        elif abs(self.pole_angle_buffer[-1])<abs(self.pole_angle_buffer[-2]):
            # returns a positive reward proportional to how closer it got the
            # pole being balanced
            return (abs(self.pole_angle_buffer[-1])/45)
        # if the action didn't change the angle
        else:
            return 0
        
        
    def state_transition(
            self,
            agent_action
            ):
        # update the cart's state as per the agent's action (Force on cart)
        friction_force=self.friction*self.cart_velocity
        cart_acceleration=(agent_action-friction_force)/self.cart_mass
        self.cart_x_position=self.cart_x_position+(self.cart_velocity*self.time_delta)
        self.cart_velocity=self.cart_velocity+(cart_acceleration*self.time_delta)
        
        # update the pole's state
        pole_angular_acceleration=(self.gravity/self.pole_length)*math.sin(self.pole_angle)
        self.pole_angle=self.pole_angle+(pole_angular_acceleration*self.time_delta)
        self.pole_velocity=self.pole_velocity+(pole_angular_acceleration*self.time_delta)
        
        # update the buffers with the new state
        self.update_buffer(cart_position=self.cart_x_position, 
                           cart_velocity=self.cart_velocity,
                           pole_angle=self.pole_angle,
                           pole_velocity=self.pole_velocity
                           )
        
        return {"cart position": self.cart_x_position, 
                "cart velocity": self.cart_velocity, 
                "pole angle": self.pole_angle, 
                "pole velocity": self.pole_velocity
                }
        
        
    


                

        