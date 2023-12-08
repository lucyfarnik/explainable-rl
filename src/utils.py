import torch as T
import torch.nn.functional as F
from collections import deque

class MovingAverage():
    """
        Keeps a list of the last n values, and overwrites the oldest value if the list is full

        Allows the user to query the simple moving average of the list,
        as well as the exponentially weighted moving average.
    """
    def __init__(self, size: int = 10, ema_alpha: float = 0.1,
                 disable_simple_average: bool = False,
                 disable_exp_average: bool = False):
        """
            Initializes the MovingAverage object.

            Args:
                size (int): The number of values to keep track of.
                ema_alpha (float): The alpha value to use for the exponentially weighted moving average.
                disable_simple_average (bool): If True, the simple moving average will be disabled
                    (and the class will work slightly faster and use less RAM).
                disable_exp_average (bool): If True, the exponentially weighted moving average will be disabled
                    (and the class will work slightly faster and use less RAM).
        """
        self.disable_simple_average = disable_simple_average
        self.disable_exp_average = disable_exp_average

        if not disable_simple_average:
            self.size = size
            self.values = deque(maxlen=size)
            self.sum = 0
        
        if not disable_exp_average:
            self.ema = None
            self.ema_alpha = ema_alpha

    def append(self, value: float):
        """
        Adds a new value to the list, overwriting the oldest value if the list is full.
        """
        if not self.disable_simple_average:
            if len(self.values) == self.size:
                self.sum -= self.values.popleft()
            self.values.append(value)
            self.sum += value

        if not self.disable_exp_average:
            if self.ema is None:
                self.ema = value
            else:
                self.ema = self.ema_alpha * value + (1 - self.ema_alpha) * self.ema

    def average(self) -> float:
        """
        Returns the average of the values in the list.
        """
        if self.disable_simple_average:
            raise Exception("Simple average is disabled — you specified this \
                            when initializing the MovingAverage object.")

        if not self.values:
            return 0
        return self.sum / len(self.values)

    def exp_average(self) -> float:
        """
        Returns the exponentially weighted moving average of the values in the list.
        """
        if self.disable_exp_average:
            raise Exception("Exponential average is disabled — you specified this \
                            when initializing the MovingAverage object.")

        if self.ema is None:
            return 0
        return self.ema



    
