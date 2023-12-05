from dataclasses import dataclass


@dataclass
class Parameter:
    """A dataclass for structuring the user adjustable parameters of an environment.

    Attributes:
        name (str): The name of the parameter. e.g. "Mass".
        default (float): The default value of the parameter. e.g. 0.1.
        min (float): The minimum value of the parameter. e.g. 0.0.
        max (float): The maximum value of the parameter. e.g. 1.0.
        unit (str): The unit of the parameter. e.g. "kg".
        lib_ref (str): Sometimes the name of the parameter in the environment is different to the name used in the streamlit app.
            This optional parameter acts as a lookup. It means the values of parameters in a specific Episode can be recovered.
            E.g.
            for parameter in parameters:
                value = env.get_attr(parameter.lib_ref)
    """

    name: str
    default: float
    min: float
    max: float
    unit: str
    lib_ref: str = None

    @property
    def key(self) -> str:
        """E.g. maze_size"""
        return self.name.replace(" ", "_").lower()

    @property
    def name_with_unit(self) -> str:
        """E.g. "Gravity (m/s^2)"""
        return f"{self.name} ({self.unit})"

    def get_lib_ref(self) -> str:
        """A getter function for the lib_ref attribute.
            If the environment and the app use the same name for the parameter,
            then the lib_ref attribute is not specified and this returns the key.

        Returns:
            str: The lib_ref attribute or the key. E.g. "masspole" or "mass".
        """
        return self.lib_ref or self.key
