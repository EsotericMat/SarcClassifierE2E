from configs.manager import ConfigManager
from sarcasm_classifier.components.preprocess import Preprocess
from sarcasm_classifier.components.train import Trainer


# Preprocessing
print('[-][-][-][-][-][-][-][-] Step 1: Preprocessing [-][-][-][-][-][-][-][-]')
preprocessing = Preprocess()
preprocessing.run()

# ## Train A model
print('[-][-][-][-][-][-][-][-] Step 2: Train A Classifier [-][-][-][-][-][-][-][-]')
trainer = Trainer()
trainer.run(target='sarcasm')