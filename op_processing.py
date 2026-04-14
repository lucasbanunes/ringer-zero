import pandas as pd
import numpy as np
import sys
import os
import pprint
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from ringer_zero.decorators import Reference
from ringer_zero import logger
import json

import op_no_pileup as op_np

# Settings
# et_list = [3, 4, 5, 6, 7]
# eta_list = [0, 1, 2, 3, 4]
et_list = [5]
eta_list = [0]
sort_list = [i for i in range(10)]
init_list = [i for i in range(5)]

for et in et_list:
    for eta in eta_list:
        
        print("="*50)
        print(f"Processing ET: {et}, Eta: {eta}")
        print("="*50)
        
        # Data quantizer
        data_file = f"qt_data_mc21_5m/2sigma_h5/mc21_13p6TeV.Zee.JF17.2sigma.5M.et{et}_eta{eta}.h5"
        print(f"Data file: {data_file}")

        seed = 512
        sort = 5
        cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        data, data_val, target, target_val = op_np.data_loader(data_file, cv, sort)
        data_all = np.concatenate((data, data_val), axis=0)

        print("="*50)

        for sort in sort_list:
            for init in init_list:

                # Model
                model_dir = f"vqat-b22-i7/et{et}-eta{eta}/tuned.vqat-b22-i7_noluts.sort_{sort}.init_{init}.model/model.tf/"
                if not os.path.exists(model_dir):
                    print(f"Directory not found: {model_dir}")
                    print("Please run this script from the project root (ringer-zero)")
                    sys.exit(1)

                # Operating Point Calculator
                linear_model = tf.keras.layers.TFSMLayer(model_dir, call_endpoint='serving_default')
                x_input = tf.keras.Input(shape=(data_all.shape[1],))
                logits_dict = linear_model(x_input)
                logits = list(logits_dict.values())[0]

                # Add sigmoid
                output = tf.keras.layers.Activation('sigmoid')(logits)
                sigmoid_model = tf.keras.Model(inputs=x_input, outputs=output)

                kw = {
                    "model": sigmoid_model, 
                    "data": (data, target), 
                    "data_val": (data_val, target_val)
                }

                history = {}

                ref_path = "references/mc21_13p6TeV.Run3_v1.40bins.ref.json"
                json_path = "op_points"
                os.makedirs(json_path, exist_ok=True)
                references = json.load(open(ref_path,'r'))
                print(references[et][eta])
                decorator = Reference(references[et][eta])
                decorator(history, kw)

                op_np.op_calculator(history, et, eta, init, sort, json_path)


