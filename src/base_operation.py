from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseOperation(ABC):
    @abstractmethod
    def transform(self, data: Any) -> Optional[Any]:
        """
        Abstract method for transforming data.
        Must be implemented in inheriting classes.

        :param data: Input data to be transformed.
        :return: Transformed data.
        """
        pass
