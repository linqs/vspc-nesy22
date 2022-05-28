#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot

analyze = __import__('analyze-results')

DATASETS_ORDER = [
    'mnist',
    'emnist',
    'fmnist',
    'kmnist',
    'mnist,emnist',
    'mnist,fmnist',
    'mnist,kmnist',
    'emnist,fmnist',
    'emnist,kmnist',
    'fmnist,kmnist',
    'emnist,fmnist,kmnist',
    'mnist,fmnist,kmnist',
    'mnist,emnist,fmnist',
    'mnist,emnist,fmnist,kmnist',
]

METHOD_ORDER = [
    'baseline-digit',
    'baseline-visual',
    'neupsl',
]

# Create graphs of accuracy per dataset on the simple setting.
def createSimpleDatasetGraphs(resultsPath, ourDir):
    headers, rows = analyze.fetchQuery('GRAPH_SIMPLE_DATASETS', resultsPath)
    headerIndexes = {headers[i]: i for i in range(len(headers))}

    #{datasets: {method: [(numPuzzles, accuracy, AUROC), ...], ...}, ...}
    data = {}

    for i in range(len(rows)):
        row = rows[i]

        method = row[headerIndexes['method']]
        datasets = row[headerIndexes['datasets']]

        if (datasets not in data):
            data[datasets] = {}

        if (method not in data[datasets]):
            data[datasets][method] = []

        data[datasets][method].append([
            row[headerIndexes['numTrain']],
            row[headerIndexes['digitAccuracy_mean']],
            row[headerIndexes['digitAccuracy_std']],
            row[headerIndexes['puzzleAccuracy_mean']],
            row[headerIndexes['puzzleAccuracy_std']],
            row[headerIndexes['puzzleAUROC_mean']],
            row[headerIndexes['puzzleAUROC_std']],
        ])

    figure, axis = matplotlib.pyplot.subplots(len(data), 3)
    row = -1

    for datasets in DATASETS_ORDER:
        if (datasets not in data):
            continue
        row += 1

        axis[row, 0].set_ylabel(datasets, rotation = 0, labelpad = 30.0, fontsize = 'x-large')

        axis[row, 0].set_ylim(0.0, 1.0)
        axis[row, 1].set_ylim(0.4, 1.0)
        axis[row, 2].set_ylim(0.4, 1.0)

        if (row == 0):
            axis[row, 0].set_title('Digit Accuracy', loc = 'center', fontsize = 'x-large')
            axis[row, 1].set_title('Puzzle Accuracy', loc = 'center', fontsize = 'x-large')
            axis[row, 2].set_title('Puzzle AUROC', loc = 'center', fontsize = 'x-large')

        methods = []
        for method in METHOD_ORDER:
            if (method not in data[datasets]):
                continue
            methods.append(method)

            numPuzzles = []
            digitAccuracy = []
            digitAurracyStd = []
            puzzleAccuracy = []
            puzzleAurracyStd = []
            auroc = []
            aurocStd = []

            for dataRow in data[datasets][method]:
                numPuzzles.append(dataRow[0])
                digitAccuracy.append(dataRow[1])
                digitAurracyStd.append(dataRow[2])
                puzzleAccuracy.append(dataRow[3])
                puzzleAurracyStd.append(dataRow[4])
                auroc.append(dataRow[5])
                aurocStd.append(dataRow[6])

            axis[row, 0].plot(numPuzzles, digitAccuracy, label = method)
            axis[row, 1].plot(numPuzzles, puzzleAccuracy, label = method)
            axis[row, 2].plot(numPuzzles, auroc, label = method)

        axis[row, 1].legend(methods)


    matplotlib.pyplot.show()

def main(resultsPath, outDir):
    createSimpleDatasetGraphs(resultsPath, outDir)

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 2 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <results path> <out dir>" % (executable), file = sys.stderr)
        sys.exit(1)

    resultsPath = args.pop(0)
    if (not os.path.isfile(resultsPath)):
        raise ValueError("Can't find the specified results path: " + resultsPath)

    outDir = args.pop(0)

    return resultsPath, outDir

if (__name__ == '__main__'):
    main(*_load_args(sys.argv))
