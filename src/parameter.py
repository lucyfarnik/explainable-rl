from dataclasses import dataclass


@dataclass
class Parameter:
    name: str
    default: float
    min: float
    max: float
    unit: str
    lib_ref: str = None

    @property
    def key(self) -> str:
        return self.name.replace(" ", "_").lower()

    @property
    def name_with_unit(self) -> str:
        return f"{self.name} ({self.unit})"

    def get_lib_ref(self) -> str:
        return self.lib_ref or self.key
