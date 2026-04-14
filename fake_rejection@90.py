import pandas as pd
import numpy as np
import sys
import os
import pprint
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from ringer_zero.decorators import Reference
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

import op_no_pileup as op_np

# Settings
# et_list = [3, 4, 5, 6, 7]
# eta_list = [0, 1, 2, 3, 4]
et_list = [5]
eta_list = [0]
pid_list = ['vloose', 'loose', 'medium', 'tight']
sort_list = [i for i in range(10)]
init_list = [i for i in range(5)]


for et in et_list:
    for eta in eta_list:
        
        print("="*50)
        print(f"Processing ET: {et}, Eta: {eta}")
        print("="*50)
        
        # Data
        data_file = f"qt_data_mc21_5m/2sigma_h5/mc21_13p6TeV.Zee.JF17.2sigma.5M.et{et}_eta{eta}.h5"
        print(f"Data file: {data_file}")

        seed = 512
        sort = 5
        cv = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        data, data_val, target, target_val = op_np.data_loader(data_file, cv, sort)

        data_all = np.concatenate((data, data_val), axis=0)
        target_all = np.concatenate((target, target_val), axis=0)

        print("="*50)

        for sort in sort_list:
            for init in init_list:

                # Model
                model_dir = f"vqat-b22-i7/et{et}-eta{eta}/tuned.vqat-b22-i7_noluts.sort_{sort}.init_{init}.model/model.tf/"
                if not os.path.exists(model_dir):
                    print(f"Directory not found: {model_dir}")
                    print("Please run this script from the project root (ringer-zero)")
                    sys.exit(1)

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

                # Predict
                output_sigmoid = sigmoid_model.predict(data_all)

                # ROC Curve
                fpr, tpr, thresholds = roc_curve(target_all, output_sigmoid)
                signal_efficiency = tpr
                fake_rejection = 1 - fpr
                target_efficiency = 0.90
                idx = np.argmin(np.abs(signal_efficiency - target_efficiency))
                target_fake_reject = fake_rejection[idx]
                print(f"Fake Rejection @ 90% Signal Efficiency: {target_fake_reject:.4f}")

                # Plot
                fig, ax = plt.subplots(figsize=(8, 7))

                # ROC Curve
                ax.plot(signal_efficiency, fake_rejection, color='green', linewidth=2, label='17x3')

                # Operating point
                ax.scatter(signal_efficiency[idx], fake_rejection[idx],
                        s=120, zorder=5, edgecolors='black', color='green',
                        label=f'@ 90% signal efficiency (FR = {target_fake_reject:.4f})')

                ax.axhline(y=target_fake_reject, color='green', linestyle='--', alpha=0.5)
                ax.axvline(x=signal_efficiency[idx], color='green', linestyle='--', alpha=0.5)

                # for pidname in pid_list:
                #     json_path = f"op_points/et{et}_eta{eta}/sort{sort}_init{init}/op_point_metrics_{key}_et{et}_eta{eta}_sort{sort}_init{init}.json"
                #     with open(json_path, 'r') as f:
                #         op = json.load(f)

                #     # Threshold
                #     idx_op = np.argmin(np.abs(thresholds - op['threshold_op']))

                #     ax.scatter(signal_efficiency[idx_op], fake_rejection[idx_op],
                #             s=120, zorder=5, edgecolors='black', color='red',
                #             label=f'Threshold ({pidname}) = {op["threshold_op"]:.4f}')

                #     ax.axhline(y=fake_rejection[idx_op], color='red', linestyle='--', alpha=0.5)
                #     ax.axvline(x=signal_efficiency[idx_op], color='red', linestyle='--', alpha=0.5)
                    
                ax.set_xlabel('Signal Efficiency', fontsize=13)
                ax.set_ylabel('Fake Rejection', fontsize=13)
                ax.set_xticks(np.arange(0, 1.2, 0.1))
                ax.set_yticks(np.arange(0, 1.2, 0.1))
                ax.set_xlim(0, 1.1)
                ax.set_ylim(0, 1.1)
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend(fontsize=11)

                plt.tight_layout()
                plots_dir = 'fake_rejection@90'
                if not os.path.exists(plots_dir):
                    os.mkdir('fake_rejection@90')
                plt.savefig(f'{plots_dir}/roc_et{et}_eta{eta}_init{init}_sort{sort}.png', dpi=150)
                print(f"Plot saved: roc_et{et}_eta{eta}_init{init}_sort{sort}.png")