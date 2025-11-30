import xgboost as xgb
import numpy as np
from configs.manager import ConfigManager
from sarcasm_classifier.components.preprocess import Preprocess
from sarcasm_classifier.components.train import Trainer


# Preprocessing
print('[-][-][-][-][-][-][-][-] Step 1: Preprocessing [-][-][-][-][-][-][-][-]')
preprocessing = Preprocess()
# preprocessing.run()
signal = preprocessing.run_single_text(text="May I have your attention please?? I mean WTF!!", add_punct=False)
# print(signal)

# ## Train A model
# print('[-][-][-][-][-][-][-][-] Step 2: Train A Classifier [-][-][-][-][-][-][-][-]')
# trainer = Trainer()
# trainer.run(target='sarcasm')

bst = xgb.Booster()
bst.load_model('models/sarcClassifier.ubj')
features = bst.feature_names
dmatrix = xgb.DMatrix(np.array([signal], dtype=float), feature_names=features)
pred = bst.predict(dmatrix)
print(pred)
if pred > .5:
    result = 'Sarcastic'
else:
    result = "Non Sarcastic"
print(f'Prediction is {result}')