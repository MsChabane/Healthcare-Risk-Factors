from src.constant import PARAMS_PATH
import yaml


def read_params(config_path=PARAMS_PATH):
    """
    Reads a YAML configuration file and returns the configuration as a dictionary.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



if __name__ == "__main__":
    config = read_params()
    print(config)
