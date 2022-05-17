import sys
import logging 
import argparse
import os
import datetime
import matplotlib.pyplot as plt

sys.path.append("../")

from kd.data import options_factory
from kd.utils import load_pickle, config_logger
from kd.evaluation.metrics import ALGORITHMIC_METRICS, HARDWARE_METRICS, METRICS_MAPPING, METRICS_DESIRED_TENDENCY_MAPPING

parser = argparse.ArgumentParser("comparison")

parser.add_argument('--dataset', type=str, default='spiral', help='define the dataset and thus task')
parser.add_argument('--folder_paths', nargs='+', type=str, default=['EXP'], help='paths to the results')
parser.add_argument('--labels', nargs='+', type=str, default=['EXP'], help='labels for the results')
parser.add_argument('--label',  type=str, default='', help='label for the result folder')

def main(): 
    # Create a folder with respect to current time where we will save the results
    args = parser.parse_args()
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(".", args.dataset+"-"+args.label+"-"+time_string)
    os.makedirs(save_path)
    config_logger(save_path=save_path)

    logging.info("# Starting comparison #")
    logging.info("## Dataset: {} ##".format(args.dataset))
    logging.info("## Result paths: {} ##".format(args.folder_paths))
    logging.info("## Labels: {} ##".format(args.labels))
    # Load the results
    results = {
        args.labels[i]: load_pickle(path+"/results.pt")
        for i, path in enumerate(args.folder_paths)
    }

    # Start with respect to algorithmic metrics and then go to hardware metrics
    _, _, _, levels, augmentation_types = options_factory(args.dataset)
    sets = ['test']
    for i, augmentation in enumerate(augmentation_types):
        sets.extend("{}_{}".format(augmentation, level) for level in levels[i])
    for set in sets:
        logging.info("## {} ##".format(set))
        for metric in ALGORITHMIC_METRICS:
            logging.info("### {} ###".format(metric))
            datapoints = []
            present = True
            for label, result in results.items():
                if not present:
                    continue
                logging.info("#### {} ####".format(label))
                if metric not in result or set not in result[metric]:
                    logging.info("{} not in result".format(set))
                    present = False
                    break
                datapoint = result[metric][set]
                logging.info("#### {} ####".format(datapoint))
                if isinstance(datapoint, float):
                    datapoint = (datapoint, 0.0) # This stands for no standard deviation, since the result was not averaged
                datapoints.append(datapoint)
            
            if present:
                means = [d[0] for d in datapoints]
                stds = [d[1] for d in datapoints]
                fig = plt.figure(figsize=(20,5))
                plt.bar(range(len(datapoints)), means, yerr=stds)
                # Also show the text labels on top of the bars
                for i, d in enumerate(datapoints):
                    plt.text(i, d[0], "{:.2f}$\pm${:.2f}".format(d[0], d[1]))

                # Then plot the winning column in a different color with respect to the metric and if it is supposed the be smaller than the rest or bigger
                # With respect to the METRICS_DESIRED_TENDENCY_MAPPING
                if metric in METRICS_DESIRED_TENDENCY_MAPPING:
                    if METRICS_DESIRED_TENDENCY_MAPPING[metric] == 'down':
                        winning_index = means.index(min(means))
                    else:
                        winning_index = means.index(max(means))
                    winner = datapoints[winning_index]
                    plt.bar([winning_index], [winner[0]], color='r', yerr=winner[1])
                plt.ylabel(METRICS_MAPPING[metric])
                plt.xticks(range(len(datapoints)), list(results.keys()))
                plt.grid()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, "{}_{}.pdf".format(set, metric)))
                plt.close(fig)
                plt.clf()

    for metric in HARDWARE_METRICS:
        logging.info("### {} ###".format(metric))
        datapoints = []
        present = True
        for label, result in results.items():
            if not present:
                continue
            logging.info("#### {} ####".format(label))
            if metric not in result:
                logging.info("{} not in result".format(metric))
                present = False
                break
            datapoint = result[metric]
            logging.info("#### {} ####".format(datapoint))
            if isinstance(datapoint, float):
                datapoint = (datapoint, 0.0)
            if metric == "flops":
                datapoint = (datapoint[0]/1e9, datapoint[1]/1e9)
            datapoints.append(datapoint)
        if present:
            fig = plt.figure(figsize=(20,5))
            means = [d[0] for d in datapoints]
            stds = [d[1] for d in datapoints]
            plt.bar(range(len(datapoints)), means, yerr=stds)
            # Also show the text labels on top of the bars
            for i, d in enumerate(datapoints):
                plt.text(i, d[0], "{:.2f}$\pm${:.2f}".format(d[0], d[1]))
            
            # Then plot the winning column in a different color with respect to the metric and if it is supposed the be smaller than the rest or bigger
            # With respect to the METRICS_DESIRED_TENDENCY_MAPPING
            if metric in METRICS_DESIRED_TENDENCY_MAPPING:
                if METRICS_DESIRED_TENDENCY_MAPPING[metric] == 'down':
                    winning_index = means.index(min(means))
                else:
                    winning_index = means.index(max(means))
                winner = datapoints[winning_index]
                plt.bar([winning_index], [winner[0]], color='r', yerr=winner[1])
            plt.ylabel(METRICS_MAPPING[metric])
            plt.xticks(range(len(datapoints)), list(results.keys()))
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "{}.pdf".format(metric)))
            plt.close(fig)
            plt.clf()

if __name__ == "__main__":
    main()