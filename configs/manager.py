from dataclasses import make_dataclass, field
from sarcasm_classifier.utils.tools import read_yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, pipe):
        self.pipe = pipe
        try:
            config_path = Path('configurations.yaml')
            self.config_dict = read_yaml(config_path)
        except:
            config_path = Path('../configurations.yaml')
            self.config_dict = read_yaml(config_path)
            print('running from module')
        self.params = self.config_dict[self.pipe]
        self.name = self.params['step_name']
        self.config = self.make_config_class()

    def make_config_class(self):
        return make_dataclass(self.name, [(k, type(v), field(default=v)) for k, v in self.params.items()])


if __name__ == '__main__':
    cm = ConfigManager('model-training').config
    # conf = cm.make_config_class()
    print(cm.train_files_path)
