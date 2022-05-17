import sys
import argparse
import numpy as np
import datetime
import logging
import os 

sys.path.append("../")

from kd.utils import load_pickle, save_pickle, config_logger

parser = argparse.ArgumentParser("average_results")
parser.add_argument('--folder_paths', nargs='+', default='EXP', help='experiment name')
parser.add_argument('--label', type=str, default='', help='label for the experiment')

def get_dict_path(dictionary, path=[]):
    for key, value in dictionary.items():
        if type(value) is dict:
            return get_dict_path(dictionary[key], path+[key])
        return path+[key]
    return path
                
def get_dict_value(dictionary, path=[], delete =True):
    if len(path)==1:
        val = dictionary[path[0]]
        if delete:
            dictionary.pop(path[0])
        return val
    else:
        return get_dict_value(dictionary[path[0]], path[1:])

def set_dict_value(dictionary, value, path=[]):
    if len(path)==1:
        dictionary[path[0]] = value
    else:
        if not path[0] in dictionary:
            dictionary[path[0]] = {}
        set_dict_value(dictionary[path[0]], value, path[1:])

def main():
    args = parser.parse_args()
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(".",args.label+"-"+time_string)
    os.makedirs(save_path)
    config_logger(save_path=save_path)

    logging.info('# Beginning analysis #')
    final_results = {}
    logging.info('## Loading of result pickles for the experiment ##')

    results = []
    if len(args.folder_paths)==1:
        args.folder_paths = args.folder_paths[0].split(" ")

    for folder_path in args.folder_paths:
        result = load_pickle(folder_path+"/results.pt")
        logging.info("### Loading result path: {} ###".format(folder_path))
        logging.info('### Loading result: {} ###'.format(result))
        results.append(result)

    assert len(results)>1

    traversing_result = results[0]
    while len(get_dict_path(traversing_result))!=0:
        path = get_dict_path(traversing_result)
        values = []
        mean = None 
        std = None 
        for result in results:
            try:
                val = get_dict_value(result, path)
                if not isinstance(val, dict):
                    values.append(val)
            except Exception as e:
                val = None
                logging.info('### Error: {} ###'.format(e))
            
        if len(values) == 0 or type(values[0]) == str or len(values)!=len(results):
            continue

        values = np.array(values)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        set_dict_value(final_results, (mean, std), path)
    
    logging.info('## Results: {} ##'.format(final_results))
    save_pickle(final_results, save_path+"/results.pt", True)
    logging.info('# Finished #')

if __name__ == '__main__':
  main()