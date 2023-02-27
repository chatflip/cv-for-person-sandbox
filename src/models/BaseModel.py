from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, input: npt.NDArray[np.uint8]) -> Any:
        pass

    @abstractmethod
    def inference(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def postprocess(
        self, input: npt.NDArray[np.uint8], output: Any
    ) -> npt.NDArray[np.uint8]:
        pass
