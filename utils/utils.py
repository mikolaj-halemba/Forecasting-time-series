import datetime
from typing import Callable, Dict, Any, Tuple
import yaml
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    :param path: Path to the YAML configuration file.
    :return: Dictionary containing the configuration parameters.
    """
    try:
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The file at path {path} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error: An error occurred while parsing the YAML file at path {path}: {exc}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return {}


def log_results(file_path: str = load_config(os.path.join('config', 'config.yaml'))['logs']['path']) -> Callable:
    """
    Decorator that logs the results of a function to a file.

    :param file_path: Path to the log file where results will be written.
    :return: Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Tuple[Dict[str, Any], float]:
            best_params, best_score = func(*args, **kwargs)
            with open(file_path, 'a') as file:
                file.write(f"Time: {datetime.datetime.now()}\n")
                file.write(f"Best Parameters: {best_params}\n")
                file.write(f"Best Score: {best_score}\n")
                file.write("\n")
            return best_params, best_score
        return wrapper
    return decorator


def log_execution_time(method):
    """
    Decorator that logs the execution time of a method.

    :param method: Method to be decorated.
    :return: Wrapped method that logs its execution time.
    """
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Executing {method.__qualname__} took {execution_time:.4f} seconds")
        return result
    return timed
