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
    'baseline-visual',
    'baseline-digit',
    'neupsl',
]

MODEL_ORDER = [
    'Baseline Visual',
    'Baseline Digit',
    'NeuPSL',
]

GRAPH_FULL_SIMPLE = 'FULL_SIMPLE'
GRAPH_OVERLAP = 'OVERLAP'
GRAPH_CROSS = 'CROSS'
GRAPHS = [GRAPH_FULL_SIMPLE, GRAPH_OVERLAP, GRAPH_CROSS]

GRAPH_OVERLAP_FILENAME = 'overlap-bars.png'
GRAPH_CROSS_FILENAME = 'cross-line.png'

COLORS = {
    'Baseline Visual': 'forestgreen',
    'baseline-visual': 'forestgreen',
    'Baseline Digit': 'tomato',
    'baseline-digit': 'tomato',
    'NeuPSL': 'mediumslateblue',
    'neupsl': 'mediumslateblue',
}

BAR_WIDTH = 6
GAP_BETWEEN_GROUPS = 5
GAP_BETWEEN_SUBGROUPS = 2
BAR_EDGE_COLOR = 'black'

# Create line graphs of dataset x dataset.
def createCrossGraphs(resultsPath, outDir):
    headers, rows = analyze.fetchQuery('GRAPH_CROSS', resultsPath)
    headerIndexes = {headers[i]: i for i in range(len(headers))}

    DATA_SOURCE_ORDER = [
        'MNIST',
        'EMNIST',
        'FMNIST',
        'KMNIST',
    ]

    # Collect all the points.
    # {(index1, index2): {method: {x: y, ...}, ...}, ...}
    points = {}
    for row in rows:
        datasets = row[headerIndexes['Data Source']].split(',')
        model = row[headerIndexes['Model']]
        numTrain = row[headerIndexes['numTrain']]
        auroc = row[headerIndexes['puzzleAUROC_mean']]
        std = row[headerIndexes['puzzleAUROC_std']]

        if (len(datasets) > 2):
            continue

        dataset1Index = DATA_SOURCE_ORDER.index(datasets[0])

        dataset2Index = dataset1Index
        if (len(datasets) == 2):
            dataset2Index = DATA_SOURCE_ORDER.index(datasets[1])

        # Use only the bottom triangle.
        indexes = tuple(sorted((dataset1Index, dataset2Index), reverse = True))

        if (indexes not in points):
            points[indexes] = {}

        if (model not in points[indexes]):
            points[indexes][model] = {}

        points[indexes][model][numTrain] = auroc

    figure, axis = matplotlib.pyplot.subplots(len(DATA_SOURCE_ORDER), len(DATA_SOURCE_ORDER), figsize = (20, 5.50))

    for (indexes, data) in points.items():
        for model in data:
            xs = list(sorted(data[model]))
            ys = [data[model][x] for x in xs]

            axis[indexes].plot(xs, ys, label = model, color = COLORS[model])

        axis[indexes].set_ylim(0.4, 0.9)

    # Set titles (diag and left) and axis labels.
    for i in range(len(DATA_SOURCE_ORDER)):
        axis[i, i].set_title(DATA_SOURCE_ORDER[i], fontsize = 'x-large', loc = 'center')
        axis[i, 0].set_title(DATA_SOURCE_ORDER[i], fontsize = 'x-large', loc = 'left', x = -0.45, y = 0.25)
        axis[i, 0].set_ylabel('AuROC', fontsize = 'large')

        axis[(len(DATA_SOURCE_ORDER) - 1), i].set_xlabel('Number of Puzzles', fontsize = 'large')

    for i in range(len(DATA_SOURCE_ORDER)):
        for j in range(len(DATA_SOURCE_ORDER)):
            if (i < j):
                figure.delaxes(axis[i, j])
            else:
                # axis[i, j].tick_params('y', 'both', False, length = 1)
                axis[i, j].tick_params(length = 2, labelsize = 8.0)

    axis[0, 0].legend(bbox_to_anchor = (4.50, 1.01), loc = 'upper right', fontsize = 'x-large')

    figure.tight_layout(h_pad = 2.0)

    matplotlib.pyplot.savefig(os.path.join(outDir, GRAPH_CROSS_FILENAME), bbox_inches = 'tight', transparent = True, pad_inches = 0.1)
    matplotlib.pyplot.show()

# Create graphs of overlap per dataset and method.
def createOverlapGraphs(resultsPath, outDir):
    headers, rows = analyze.fetchQuery('GRAPH_OVERLAP', resultsPath)
    headerIndexes = {headers[i]: i for i in range(len(headers))}

    DATA_SOURCE_ORDER = [
        'MNIST',
        'EMNIST',
        'FMNIST',
        'KMNIST',
    ]

    NUM_UNIQUE_IMAGES_ORDERING = [80, 160, 320]
    NUM_PUZZLE_ORDERING = {
        80: [-1, 5, 10],
        160: [10, 20, 30],
        320: [20, 30, 40],
    }

    # 9 bars, 2 spaces between the groups, and 2 end caps.
    IMAGE_GROUP_SIZE = 9 * BAR_WIDTH + 2 * GAP_BETWEEN_SUBGROUPS + 2 * GAP_BETWEEN_GROUPS

    # 3 bars, 1 space between the groups.
    PUZZLE_GROUP_SIZE = 3 * BAR_WIDTH + GAP_BETWEEN_SUBGROUPS

    INITIAL_SPACE = -5

    figure, axis = matplotlib.pyplot.subplots(len(DATA_SOURCE_ORDER), 1, figsize = (20, 7.5))

    # We know the exact position of each bar/row as it comes out, so no need for pre-processing.

    # {dataset: {model: handle, ...}, ...}
    legendHandles = {dataset: {} for dataset in DATA_SOURCE_ORDER}

    xTicksMajorPositions = []
    xTicksMajorLabels = []

    xTicksMinorPositions = []
    xTicksMinorLabels = []

    for row in rows:
        model = row[headerIndexes['Model']]
        datasets = row[headerIndexes['Data Source']]
        numUniqueImages = int(row[headerIndexes['Number of Unique Images']])
        numPuzzles = row[headerIndexes['Number of Puzzles']]
        auroc = row[headerIndexes['puzzleAUROC_mean']]
        std = row[headerIndexes['puzzleAUROC_std']]

        datasetsIndex = DATA_SOURCE_ORDER.index(datasets)
        imagesIndex = NUM_UNIQUE_IMAGES_ORDERING.index(numUniqueImages)
        puzzlesIndex = NUM_PUZZLE_ORDERING[numUniqueImages].index(numPuzzles)
        modelIndex = MODEL_ORDER.index(model)

        position = INITIAL_SPACE \
                + imagesIndex * IMAGE_GROUP_SIZE \
                + puzzlesIndex * PUZZLE_GROUP_SIZE \
                + modelIndex * BAR_WIDTH

        if (datasetsIndex == 0 and modelIndex == 1):
            xTicksMinorPositions.append(position)
            xTicksMinorLabels.append(numPuzzles)

            if (puzzlesIndex == 1):
                # Major ticks will hide minor if they have the same position.
                xTicksMajorPositions.append(position + 0.01)
                xTicksMajorLabels.append("Number of Puzzles\n$\\mathbf{%d \\, Unique \\, Images}$" % (numUniqueImages))

        bar = axis[datasetsIndex].bar(x = position, width = BAR_WIDTH,
                height = auroc, yerr = std,
                color = COLORS[model], edgecolor = BAR_EDGE_COLOR)

        if (model not in legendHandles[datasets]):
            legendHandles[datasets][model] = bar

    for i in range(len(DATA_SOURCE_ORDER)):
        # An empty bar to put in an empty space at the beginning.
        axis[i].bar(x = 0, width = 0.01, height = 0)

        axis[i].set_ylim(0.4, 0.9)
        axis[i].set_ylabel('AuROC')

        axis[i].set_title(DATA_SOURCE_ORDER[i], fontsize = 'x-large', loc = 'left', x = -0.080, y = 0.25)

        handles = []
        labels = []
        for (model, handle) in legendHandles[DATA_SOURCE_ORDER[i]].items():
            labels.append(model)
            handles.append(handle)

        axis[i].legend(handles, labels, loc = 'center left')

        axis[i].set_xticks(ticks = xTicksMinorPositions, labels = xTicksMinorLabels, minor = True)
        axis[i].tick_params(axis = 'x', which = 'minor', size = 1)
        axis[i].set_xticks(ticks = xTicksMajorPositions, labels = xTicksMajorLabels, minor = False)
        axis[i].tick_params(axis = 'x', which = 'major', pad = 20, size = 0)

        # Put in vertical lines.
        # HACK(eriq): Position is strange.
        for j in range(1, 3):
            axis[i].axvline(x = INITIAL_SPACE + (j * IMAGE_GROUP_SIZE) - 8, color = 'black')

    figure.tight_layout(h_pad = 1)
    matplotlib.pyplot.savefig(os.path.join(outDir, GRAPH_OVERLAP_FILENAME), bbox_inches = 'tight', transparent = True, pad_inches = 0.1)
    matplotlib.pyplot.show()

# Create graphs of accuracy per dataset on the simple setting.
def createSimpleDatasetGraphs(resultsPath, outDir):
    headers, rows = analyze.fetchQuery('GRAPH_SIMPLE_DATASETS', resultsPath)
    headerIndexes = {headers[i]: i for i in range(len(headers))}

    # {datasets: {method: [(numPuzzles, accuracy, AUROC), ...], ...}, ...}
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

def main(resultsPath, outDir, graph):
    os.makedirs(outDir, exist_ok = True)

    if (graph == GRAPH_FULL_SIMPLE):
        createSimpleDatasetGraphs(resultsPath, outDir)
    elif (graph == GRAPH_OVERLAP):
        createOverlapGraphs(resultsPath, outDir)
    elif (graph == GRAPH_CROSS):
        createCrossGraphs(resultsPath, outDir)
    else:
        raise ValueError("Unknown graph type '%s'. Expected one of [%s]." % (graph, ', '.join(GRAPHS)))

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 3 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <results path> <out dir> <graph>" % (executable), file = sys.stderr)
        sys.exit(1)

    resultsPath = args.pop(0)
    if (not os.path.isfile(resultsPath)):
        raise ValueError("Can't find the specified results path: " + resultsPath)

    outDir = args.pop(0)

    graph = args.pop(0).upper()
    if (graph not in GRAPHS):
        raise ValueError("Unknown graph type '%s'. Expected one of [%s]." % (graph, ', '.join(GRAPHS)))

    return resultsPath, outDir, graph

if (__name__ == '__main__'):
    main(*_load_args(sys.argv))
