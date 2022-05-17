import matplotlib.pyplot as plt
import matplotlib
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import numpy as np
import matplotlib.gridspec as gridspec

# Remove the frame for the legend
matplotlib.rcParams.update({'font.size': 8, 'font.family': 'serif', 'legend.frameon': False})

from kd.evaluation.utils import classification_uncertainty, regression_uncertainty
from kd.data.regress import toy_y

def total_variation(hist1, hist2):
    assert len(hist1) == len(hist2)
    return np.sum(np.abs(hist1 - hist2))

class Plotter:
    def __init__(self, path, name="", columns=2, rows=2):
        self.path = path
        self.name = name
        self.val_range = None
        self.columns =columns
        self.rows = rows
        self.cache = {}

    def plot(self, outputs, labels, split):
        pass

    def plot_final(self):
        pass 

class PlotterSpiral(Plotter):
    def plot(self, outputs, labels, split):
        cols = 3
        rows = len(labels)
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3))
        for i in range(len(labels)):
            output = outputs[i]
            label = labels[i]
            predictive, aleatoric, epistemic = classification_uncertainty(output)
            # Convert them to numpy and put them into cpu
            predictive = predictive.squeeze().cpu().numpy().reshape(int(math.sqrt(output.shape[0])), -1)
            aleatoric = aleatoric.squeeze().cpu().numpy().reshape(int(math.sqrt(output.shape[0])), -1)
            epistemic = epistemic.squeeze().cpu().numpy().reshape(int(math.sqrt(output.shape[0])), -1)

            if self.val_range is None:
                self.val_range = (0.0, math.log(output.shape[-1]))
            ax = axs[i, 0] if rows>1 else axs[0]
            im = ax.imshow(predictive, extent=[-20, 20, -20, 20],vmin=self.val_range[0], vmax=self.val_range[1],cmap='jet')
            ax.set_ylabel(label, labelpad=-1)
            ax.set_aspect('auto')
            if i == 0:
                ax.set_title("Predictive Uncertainty")
            self.cache[label] = predictive

            ax = axs[i, 1] if rows>1 else axs[1]
            im = ax.imshow(aleatoric, extent=[-20, 20, -20, 20],vmin=self.val_range[0], vmax=self.val_range[1],cmap='jet')
            ax.set_aspect('auto')
            if i == 0:
                ax.set_title("Aleatoric Uncertainty")

            ax = axs[i, 2] if rows>1 else axs[2]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(epistemic, extent=[-20, 20, -20, 20],vmin=self.val_range[0], vmax=self.val_range[1],cmap='jet')
            fig.colorbar(im, cax=cax, orientation='vertical',shrink=1.0)
            cax.set_ylabel('Entropy [nats]', rotation=270, labelpad=10.0)
            ax.set_aspect('auto')
            if i == 0:
                ax.set_title("Epistemic Uncertainty")

        fig.tight_layout()
        plt.savefig(self.path+'/'+self.name+'_'+split+'.pdf')
        plt.close(fig)
        plt.cla()
        plt.clf()

    def plot_final(self):
        # Here we collect all the predictive uncertainty and plot them together in two rows
        # the labels are going to be preserved for the axis titles
        ncols = self.columns
        nrows = self.rows
        nfigs = len(self.cache)
        assert nfigs<=nrows*ncols

        fig = plt.figure(figsize=(ncols*3.5, nrows*3.))
        m = nfigs % ncols
        m = range(1, ncols+1)[-m]  # subdivision of columns
        gs = gridspec.GridSpec(nrows, m*ncols)
        
        for i in range(len(self.cache)):
            row = i // ncols
            col = i % ncols

            if row == nrows-1 and ncols*nrows!=nfigs: # center only last row
                off = int(m * (ncols - nfigs % ncols) / 2)
            else:
                off = 0

            ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
            predictive = self.cache[list(self.cache.keys())[i]]
            im = ax.imshow(predictive, extent=[-20, 20, -20, 20],vmin=self.val_range[0], vmax=self.val_range[1],cmap='jet')
            ax.set_title(list(self.cache.keys())[i])
            # Plot the colorbar at the end of the row
            if col == ncols-1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical',shrink=1.0)
                cax.set_ylabel('Entropy [nats]', rotation=270, labelpad=10.0)
            if col == 0:   
                ax.set_ylabel("Predictive Uncertainty", labelpad=-1)
            # Set the xlabel as (a), (b), (c), ...
            ax.set_xlabel("("+chr(ord('a')+i)+")", fontsize=12)
            ax.set_aspect('auto')
        
        fig.tight_layout()
        plt.savefig(self.path+'/'+self.name+'_final.pdf')
        plt.close(fig)
        plt.cla()
        plt.clf()

class PlotterRegress(Plotter):
    def plot(self, outputs, labels, split):
        ncols = self.columns
        nrows = self.rows
        nfigs = len(outputs)
        assert nfigs<=nrows*ncols
        
        fig = plt.figure(figsize=(ncols*3, nrows*3.))
        m = nfigs % ncols
        m = range(1, ncols+1)[-m]  # subdivision of columns
        gs = gridspec.GridSpec(nrows, m*ncols)

        for i in range(len(labels)):
            output = outputs[i]
            label = labels[i]
            row = i // ncols
            col = i % ncols

            if row == nrows-1 and ncols*nrows!=nfigs: # center only last row
                off = int(m * (ncols - nfigs % ncols) / 2)
            else:
                off = 0

            ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
            ax.set_title(label)
            x = torch.linspace(-12, 12, 1000).reshape(-1, 1)
            y = toy_y(x)
            ax.plot(x.squeeze().cpu().numpy(),
                    y.squeeze().cpu().numpy(), color='red', label="Ground Truth")
            predictive, aleatoric, epistemic = regression_uncertainty(output)
            predictive = predictive.squeeze().cpu().numpy()
            aleatoric = aleatoric.squeeze().cpu().numpy()
            epistemic = epistemic.squeeze().cpu().numpy()

            if len(output.shape) == 2:
                output = output.unsqueeze(1)
            mean = output[:, :, 0].mean(axis=1).cpu().numpy()
            x = x.squeeze().cpu().numpy()
            ax.plot(x, mean,label="Mean")
            ax.fill_between(x, mean-aleatoric, mean+aleatoric,
                            alpha=0.25, color='green', label="Aleatoric Unc.")
            ax.fill_between(x, mean-epistemic, mean+epistemic,
                            alpha=0.25, color='blue', label="Epistemic Unc.")
            ax.fill_between(x, mean-predictive, mean+predictive,
                            alpha=0.25, color='purple', label="Predictive Unc.")

            ax.axis(ymin=-5,ymax=20)
            if col==0 and row==0:
                ax.legend(fontsize = 11)
            ax.set_xlabel("("+chr(ord('a')+i)+")", fontsize=12)
        fig.tight_layout()
        plt.savefig(self.path+'/'+self.name+'_'+split+'.pdf')
        plt.close(fig)
        plt.cla()
        plt.clf()

class PlotterClassification(Plotter):
    def plot(self, outputs, labels, split):
        fig, axs = plt.subplots(len(labels), 3, figsize=(13, 2*len(labels)))
        for i in range(len(labels)):
            output = outputs[i]
            label = labels[i]
            predictive, aleatoric, epistemic = classification_uncertainty(output)
            # Convert them to numpy and put them into cpu
            predictive = predictive.squeeze().cpu().numpy()
            aleatoric = aleatoric.squeeze().cpu().numpy()
            epistemic = epistemic.squeeze().cpu().numpy()

            if self.val_range is None:
                self.val_range = (0.0, math.log(output.shape[-1]))

            # Onto each axis create a histogram of predictive, aleatoric and epistemic uncertainty
            ax = axs[i, 0]
            ax.hist(predictive, bins=50, density=True, range=self.val_range, label=label, alpha=0.5)
            ax.set_title("Predictive Uncertainty")
            ax.set_ylabel(label)
            ax.set_xlabel("Entropy [nats]")

            ax = axs[i, 1]
            ax.hist(aleatoric, bins=50, density=True, range=self.val_range, label=label, alpha=0.5)
            ax.set_title("Aleatoric Uncertainty")
            ax.set_xlabel("Entropy [nats]")

            ax = axs[i, 2]
            ax.hist(epistemic, bins=50, density=True, range=self.val_range, label=label, alpha=0.5)
            ax.set_title("Epistemic Uncertainty")
            ax.set_xlabel("Entropy [nats]")
            if split not in self.cache:
                self.cache[split] = {}
            self.cache[split][label] = (predictive, aleatoric, epistemic)
        fig.tight_layout()
        plt.savefig(self.path+'/'+split+'.pdf')
        plt.close(fig)
        plt.cla()
        plt.clf()
            
    def plot_final(self):
        for i, split in enumerate(self.cache):
            # The ensemble always needs to go first!
            fig, axs = plt.subplots(1, 3, figsize=(25, 4), constrained_layout=True) 
            ensemble_predictive, ensemble_aleatoric, ensemble_epistemic = self.cache[split][list(self.cache[split].keys())[0]]
            ensemble_predictive = np.histogram(ensemble_predictive, bins=50, density=True, range=self.val_range)[0]
            ensemble_aleatoric = np.histogram(ensemble_aleatoric, bins=50, density=True, range=self.val_range)[0]
            ensemble_epistemic = np.histogram(ensemble_epistemic, bins=50, density=True, range=self.val_range)[0]
            for j, label in enumerate(self.cache[split]):
                predictive, aleatoric, epistemic = self.cache[split][label]

                ax = axs[0]
                # Compute total varation with respect to the first label which is the ensemble
                tv = total_variation(np.histogram(predictive, bins=50, range=self.val_range, density=True)[0],
                                                  ensemble_predictive)
                ax.hist(predictive, bins=50, density=True, range=self.val_range, label=label+"\nTV to Ensemble: {:.2f}".format(tv), alpha=0.5)
                ax.set_title("Predictive Uncertainty", fontsize=14)
                if j == len(self.cache[split])-1:
                    ax.set_xlabel("Entropy [nats]", fontsize=14)
                    ax.set_ylabel("Probability Density", fontsize=14)
                    ax.legend(ncol=2, fontsize=14)
                    # Show (a) under the xlabel, such that we can refer to it in the paper
                    ax.text(-0.05, 1.05, '(a)', transform=ax.transAxes, fontsize=14, verticalalignment='top')

                ax = axs[1]
                # Compute total varation with respect to the first label which is the ensemble
                tv = total_variation(np.histogram(aleatoric, bins=50, range=self.val_range, density=True)[0],
                                                  ensemble_aleatoric)
                ax.hist(aleatoric, bins=50, density=True, range=self.val_range, label="TV to Ensemble: {:.2f}".format(tv), alpha=0.5)
                if j == len(self.cache[split])-1:
                    ax.set_title("Aleatoric Uncertainty", fontsize=14)
                    ax.set_xlabel("Entropy [nats]", fontsize=14)
                    ax.legend(ncol=2, fontsize=14)
                    # Show (b) under the xlabel, such that we can refer to it in the paper
                    ax.text(-0.05, 1.05, '(b)', transform=ax.transAxes, fontsize=14, verticalalignment='top')

                ax = axs[2]
                # Compute total varation with respect to the first label which is the ensemble
                tv = total_variation(np.histogram(epistemic, bins=50, range=self.val_range, density=True)[0],
                                     ensemble_epistemic)
                ax.hist(epistemic, bins=50, density=True, range=self.val_range, label="TV to Ensemble: {:.2f}".format(tv), alpha=0.5)
                if j == len(self.cache[split])-1:    
                    ax.set_title("Epistemic Uncertainty", fontsize=14)
                    ax.set_xlabel("Entropy [nats]", fontsize=14)
                    ax.legend(ncol=2, fontsize=14)
                    # Show (c) under the xlabel, such that we can refer to it in the paper
                    ax.text(-0.05, 1.05, '(c)', transform=ax.transAxes, fontsize=14, verticalalignment='top')      

            plt.savefig(self.path+'/'+split+'_final.pdf')
            plt.close(fig)
            plt.cla()
            plt.clf()

def plotting_factory(dataset):
    if dataset == "spiral":
        return PlotterSpiral
    elif dataset == "regress":
        return PlotterRegress
    elif dataset in ["cifar", "svhn"]:
        return PlotterClassification
    else:
        raise ValueError("Dataset not supported")
