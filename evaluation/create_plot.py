# ==Imports==
import os, sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# ==============================
#  Create plots for evaluation
# ==============================

class PlotEvaluation:
    def __init__(self, args, input_path, output_path):
        self.args = args

        if not os.path.exists(input_path):
            raise FileNotFoundError("Path %s does not exist!", input_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.input_path = input_path
        self.output_path = output_path


    def single_plot(self):
        """
        Plot signals of different algorithms in single frame
        """
        signal_dict = {}

        # read signals, calculate mean and std deviation
        print("Access %s..." % self.input_path)
        for algo in self.args.algos:
            print("Read files for algorithm: %s" % algo)
            signal_dict[algo] = {}
            signal_y_values = np.array([])
            i = 0
            for filename in os.listdir(self.input_path):
                if algo in filename and ".csv" in filename:
                    print(filename)
                    data = pd.read_csv(os.path.join(self.input_path, filename)) #np.genfromtxt(os.path.join(self.input_path, filename), dtype=float, delimiter=',', names=True)
                    if i == 0:
                        print(signal_y_values)
                        signal_y_values = np.append(signal_y_values, data["Value"].to_numpy(float))
                        signal_dict[algo]["x_value"] = data["Step"].to_numpy(int)
                    else: 
                        signal_y_values = np.vstack((signal_y_values, data["Value"].to_numpy(float)))
                    
                    i += 1
                    
            signal_dict[algo]["mean"] = np.mean(signal_y_values, axis = 0)
            signal_dict[algo]["std"] = np.std(signal_y_values, axis = 0)

        # plotting
        print("create plot...")

        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('axes', labelsize=14)
        
        # width as measured in inkscape
        width = 1.5*3.487
        height = width / 1.618

        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

        if self.args.grid:
            plt.grid(linestyle='--', color='silver', which='both')
        
        for i, algo in enumerate(self.args.algos):
            if self.args.legend:
                label = self.args.legend[i]
            else: 
                label = algo
            plt.plot(signal_dict[algo]["x_value"], signal_dict[algo]["mean"], self.args.mean_color_list[i], linewidth=1.0, label=label)
            plt.fill_between(signal_dict[algo]["x_value"], signal_dict[algo]["mean"]-signal_dict[algo]["std"], signal_dict[algo]["mean"]+signal_dict[algo]["std"], facecolor=self.args.std_color_list[i], edgecolor=self.args.std_color_list[i])

        # title 
        plt.title(self.args.title)

        # axis
        ax.set_ylabel(self.args.ylabel)
        ax.set_xlabel(self.args.xlabel)
        ax.xaxis.set_major_formatter(ticker.EngFormatter())
        ax.xaxis.set_ticks(np.arange(0, max(signal_dict[self.args.algos[0]]["x_value"]), 100000), minor=True)

        # legend 
        ax.legend(bbox_to_anchor=(1, 0.3))

        # set size of figure
        fig.set_size_inches(width, height)

        # save plot
        if self.args.plotname == "":
            for algo in self.args.algos:
                self.args.plotname += algo + "_"
            self.args.plotname += "eval.jpg"

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_path, self.args.plotname), dpi=200)
        print("figure created.")


if __name__ == "__main__":
    # Paths
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plot_data'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots_created'))

    # Get input arguments from shell
    parser = argparse.ArgumentParser("Create evaluation plots")

    # General configs for Plotting
    parser.add_argument("--plotname", default="", type=str, help="Specify name of plot")
    parser.add_argument("--std", action="store_false", default=True, help="Plot standard deviation")

    parser.add_argument("--error", default="overlap_area", type=str, help="Specify which error type should be computed. Choose either euclidean, area or overlap_area")
    parser.add_argument("--number", default=10, type=int, help="Specify number of experiments that should be evaluated")

    # Configs for plot
    parser.add_argument("--title", default="Environment", type=str, help="Specify title of plot")
    parser.add_argument("--xlabel", default="x-axis", type=str, help="Specify label for x-axis")
    parser.add_argument("--ylabel", default="y-axis", type=str, help="Specify label for y-axis")
    parser.add_argument("--grid", action="store_false", default=True, help="Plot grid")
    parser.add_argument('--mean_color_list', nargs='+', type=str, default=["b-", "r-", "g-"], help='Specfiy line color for mean values')
    parser.add_argument('--std_color_list', nargs='+', type=str, default=["cornflowerblue", "lightcoral", "lightgreen"], help='Specfiy face color for std values')

    # Configs for RL-algorithms
    parser.add_argument('--algos', nargs='+', type=str, default=["rarl"], help='Specfiy list of algorithms to plot')
    parser.add_argument('--legend', nargs='+', type=str, default=[], help='Specfiy face color for std values')

    # Get arguments
    args = parser.parse_args()

    # Create Plot
    plotter = PlotEvaluation(args, data_path, output_path)
    plotter.single_plot()