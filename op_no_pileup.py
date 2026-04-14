import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from ringer_zero.decorators import Reference
from ringer_zero import logger
import json
import pandas as pd
import os
import numpy as np

def data_loader( path, cv, sort):

  pidname = 'el_lhmedium'
  df = pd.read_hdf(path)
  df = df.loc[ ((df[pidname]==True) & (df.target==1.0)) | ((df.target==0) & (df['el_lhvloose']==False) ) ]

  
  # for new training, we selected 1/2 of rings in EM layers
  #pre-sample - 8 rings
  # EM1       - 64 rings
  # EM2       - 8 rings
  # EM3       - 8 rings
  # Had1      - 4 rings
  # Had2      - 4 rings
  # Had3      - 4 rings
  prefix = 'trig_L2_cl_ring_%i'

  # rings presmaple 
  presample = [prefix %iring for iring in range(8//2)]

  # EM1 list
  sum_rings = 8
  em1 = [prefix %iring for iring in range(sum_rings, sum_rings+(64//2))]

  # EM2 list
  sum_rings = 8+64
  em2 = [prefix %iring for iring in range(sum_rings, sum_rings+(8//2))]

  # EM3 list
  sum_rings = 8+64+8
  em3 = [prefix %iring for iring in range(sum_rings, sum_rings+(8//2))]

  # HAD1 list
  sum_rings = 8+64+8+8
  had1 = [prefix %iring for iring in range(sum_rings, sum_rings+(4//2))]

  # HAD2 list
  sum_rings = 8+64+8+8+4
  had2 = [prefix %iring for iring in range(sum_rings, sum_rings+(4//2))]

  # HAD3 list
  sum_rings = 8+64+8+8+4+4
  had3 = [prefix %iring for iring in range(sum_rings, sum_rings+(4//2))]
  col_names = presample+em1+em2+em3+had1+had2+had3
  # print(col_names)

  rings = df[col_names].values.astype(np.float32)

  def norm1( data ):
      norms = np.abs( data.sum(axis=1) )
      norms[norms==0] = 1
      return data/norms[:,None]
    
  target = df['target'].values.astype(np.int16)
  rings = norm1(rings)
  # print(f"Rings shape: {rings.shape}")
  # print(f"Target shape: {target.shape}")
  splits = [(train_index, val_index) for train_index, val_index in cv.split(rings,target)]
  # return 
  return rings [ splits[sort][0]], rings [ splits[sort][1]], target [ splits[sort][0] ], target [ splits[sort][1] ]


def op_calculator(history, et, eta, init, sort, json_path):

    if 'reference' in history:
            all_thresholds = []
            ordered_keys = ['vloose', 'loose', 'medium', 'tight']
            for key in ordered_keys:
                if key not in history['reference']:
                    continue
                metrics = history['reference'][key]

                print(f"\nOperating Point: {key}")
                
                # Helper to convert numpy types for JSON serialization
                def convert_to_serializable(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj

                # Save metrics to JSON
                # Create a serializable copy of metrics
                serializable_metrics = {k: convert_to_serializable(v) for k, v in metrics.items()}
                op_metrics = {k: v for k, v in serializable_metrics.items() if k.endswith('_op')}
                json_filename = f"{json_path}/et{et}_eta{eta}/sort{sort}_init{init}/op_point_metrics_{key}_et{et}_eta{eta}_sort{sort}_init{init}.json"
                os.makedirs(f"{json_path}/et{et}_eta{eta}/sort{sort}_init{init}", exist_ok=True)
                with open(json_filename, 'w') as f:
                    json.dump(op_metrics, f, indent=4)
                print(f"Saved metrics to {json_filename}")

                def print_metrics(prefix, d_metrics, suffix=''):
                    pd = d_metrics.get(f'pd{suffix}')
                    fa = d_metrics.get(f'fa{suffix}')
                    sp = d_metrics.get(f'sp{suffix}')
                    th = d_metrics.get(f'threshold{suffix}')
                    
                    pd_val = pd[0] if isinstance(pd, (tuple, list, np.ndarray)) else pd
                    fa_val = fa[0] if isinstance(fa, (tuple, list, np.ndarray)) else fa
                    th_val = th[0] if isinstance(th, (tuple, list, np.ndarray)) else th
                    
                    if suffix == '_op' and th_val is not None: # collect thresholds for operating points
                        all_thresholds.append(float(th_val))

                    print(f"  {prefix}:")
                    print(f"    PD       : {pd_val:.4%}")
                    print(f"    FA       : {fa_val:.4%}")
                    print(f"    SP       : {sp:.4f}")
                    print(f"    Threshold: {th}")

                print_metrics("Operation (Combined)", metrics, '_op')
