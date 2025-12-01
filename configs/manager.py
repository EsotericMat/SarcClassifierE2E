from dataclasses import make_dataclass, field
from sarcasm_classifier.utils.tools import read_yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, pipe):
        self.pipe = pipe
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / 'configurations.yaml'

        try:
            self.config_dict = read_yaml(config_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}. "
                f"Please ensure 'configurations.yaml' exists in the project root."
            )

        self.params = self.config_dict[self.pipe]
        self.name = self.params['step_name']
        self.config = self.make_config_class()

    def make_config_class(self):
        return make_dataclass(self.name, [(k, type(v), field(default=v)) for k, v in self.params.items()])


if __name__ == '__main__':
    cm = ConfigManager('model-training').config
    # conf = cm.make_config_class()
    print(cm.train_files_path)
