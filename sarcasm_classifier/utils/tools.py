import os
from pathlib import Path
import yaml

def read_yaml(path: Path):
    try:
        with open(path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            # logger.info(f"yaml file: {path} loaded successfully")
            return content
    except Exception as e:
        raise ValueError(f"Can't return configurations: {e}")

def validate_path(base: Path) -> Path:
    """
    Validate that the path is exist, or create it whenever needed.
    :param base: Directory
    :return: None
    """
    output_dir = Path(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def connect_data_dirs(base: Path, file: Path) -> Path:
    return os.path.join(base, file)

