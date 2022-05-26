#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot

analyze = __import__('analyze-results')

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
            row[headerIndexes['puzzleAUROC_mean']],
            row[headerIndexes['puzzleAUROC_std']],
        ])

    # TEST
    print(data)

    for datasets in data:
        plots = []

        for method in data[datasets]:
            plot = {
                'x': [],
                'y': [],
                'label': method,
            }

            for (numPuzzles, accuracy, aurracyStd, auroc, aurocStd) in data[datasets][method]:
                plot['x'].append(numPuzzles)
                plot['y'].append(auroc)

            plots.append(plot)

        ''' TEST
        for plot in plots:
            matplotlib.pyplot.plot(plot['x'], plot['y'], label = plot['label'])
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
        '''
        print(plots)

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
