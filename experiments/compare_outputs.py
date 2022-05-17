import sys
import logging 
import argparse
import os
import datetime

sys.path.append("../")

from kd.data import options_factory
from kd.utils import load_pickle, config_logger
from kd.evaluation.plotting import plotting_factory
from kd.evaluation.metrics import METRICS_MAPPING

parser = argparse.ArgumentParser("comparison")
parser.add_argument('--dataset', type=str, default='spiral', help='define the dataset and thus task')
parser.add_argument('--folder_paths', nargs='+', type=str, default=['EXP'], help='paths to the results')
parser.add_argument('--labels', nargs='+', type=str, default=['EXP'], help='labels for the results')
parser.add_argument('--label',  type=str, default='', help='label for the result folder')
parser.add_argument('--columns',  type=int, default=2, help='label for the result folder')
parser.add_argument('--rows',  type=int, default=2, help='label for the result folder')

def main():
    # Create a folder with respect to current time where we will save the results
    args = parser.parse_args()
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(".", args.dataset+"-"+args.label+"-"+time_string)
    os.makedirs(save_path)
    config_logger(save_path=save_path)

    logging.info("# Starting plotting #")
    logging.info("## Dataset: {} ##".format(args.dataset))
    logging.info("## Folder paths: {} ##".format(args.folder_paths))
    logging.info("## Labels: {} ##".format(args.labels))

    splits = ["test"]
    _, _, _, levels, augmentation_types = options_factory(args.dataset)
    for i, augmentation in enumerate(augmentation_types):
        for level in levels[i]:
            splits+= ["{}_{}".format(augmentation, level)]
    plotting_class = plotting_factory(args.dataset)
    plotting_class = plotting_class(save_path, args.dataset, args.columns, args.rows)

    main_metric = "error" if args.dataset not in ["regress"] else "nll"

    for i, split in enumerate(splits):
        logging.info("## {} ##".format(split))
        outputs = []
        labels = []
        if args.dataset in ["spiral", "regress"] and split in ["train", "valid", "test"]:
            logging.info("## {} not supported for plotting for {} ##".format(split, args.dataset))
            continue

        for j, label in enumerate(args.labels):
            # Load the output pickle
            output = load_pickle(os.path.join(args.folder_paths[j], split+"_outputs.pt"))
            # Add the output to the list
            outputs.append(output)

            results = load_pickle(os.path.join(args.folder_paths[j], "results.pt"))
            metric = results[main_metric][split]
            if args.dataset in ["spiral", "regress"]:
                metric = results[main_metric]["test"]
            if isinstance(metric, tuple):
                metric = metric[0] # Here we extract just the mean
            labels.append(label+" "+METRICS_MAPPING[main_metric]+": {:.2f}".format(metric))

        # Plot the results
        plotting_class.plot(outputs, labels, split)
    plotting_class.plot_final()

if __name__ == "__main__":
    main()