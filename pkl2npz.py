import numpy as np
import pickle
import sys, os
from os import path

if __name__=='__main__':

    assert len(sys.argv) >= 2

    output_dir = os.path.curdir
    os.makedirs(output_dir, exist_ok=True)

    for model_path in sys.argv[1:]:
        with open(model_path, 'rb') as model_file:
            try:
                model_data = pickle.load(model_file, encoding='latin1')
            except:
                model_data = pickle.load(model_file)

            output_data = {}
            for key, data in model_data.items():
                dtype = str(type(data))
                if 'chumpy' in dtype:
                    # Convert chumpy
                    output_data[key] = np.array(data)
                elif 'scipy.sparse' in dtype:
                    # Convert scipy sparse matrix
                    output_data[key] = data.toarray()
                else:
                    output_data[key] = data
            model_fname = path.split(model_path)[1]
            if len(model_fname) > 11 and model_fname[11] == 'f':
                output_gen = 'FEMALE'
            elif len(model_fname) > 11 and model_fname[11] == 'm':
                output_gen = 'MALE'
            else:
                output_gen = 'NEUTRAL'
            output_path = path.join(output_dir, 'SMPL_' + output_gen + '.npz')
            print('Writing', output_path)
            np.savez_compressed(output_path, **output_data)