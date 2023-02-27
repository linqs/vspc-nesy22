#!/usr/bin/env python3

# Generate a split of puzzles.

import argparse
import datetime
import glob
import json
import math
import os
import shutil

import pslpython.neupsl
import tensorflow

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_RAW_DATA_DIR = os.path.join(THIS_DIR, '..', 'data', 'raw')
DEFAULT_DATA_DIR = os.path.join(THIS_DIR, '..', 'data', 'vspc')

UNSUPPORTED_STRATEGIES = ['r_puzzle', 'r_cell', 'transfer']

OPTIONS_FILENAME = 'options.json'

CELL_LABELS_FILENAME = 'cell_labels.txt'
PUZZLE_PIXELS_FILENAME = 'puzzle_pixels.txt'
PUZZLE_LABELS_FILENAME = 'puzzle_labels.txt'
PUZZLE_NOTES_FILENAME = 'puzzle_notes.txt'

LABEL_MAP_FILENAME = 'label_id_map.txt'

UNTRAINED_DIGIT_MODEL_H5_FILENAME = 'digit_model_untrained.h5'
UNTRAINED_DIGIT_MODEL_TF_DIRNAME = 'digit_model_untrained_tf'

DIGIT_FEATURES_FILENAME = 'digit_features.txt'
DIGIT_LABELS_FILENAME = 'digit_labels.txt'
DIGIT_TARGETS_FILENAME = 'digit_targets.txt'
DIGIT_TRUTH_FILENAME = 'digit_truth.txt'

DIGIT_PINNED_TRUTH_FILENAME = 'positive_digit_pinned_truth.txt'

PUZZLE_POSITIVE_FIRST_ID_FILENAME = 'first_positive_puzzle.txt'

ROW_VIOLATIONS_FILENAME = 'row_col_violation_targets.txt'
VIOLATIONS_TARGETS_FILENAME = 'violation_targets.txt'
VIOLATIONS_TRUTH_FILENAME = 'violation_truth.txt'

ROW_VIOLATIONS_POSITIVE_FILENAME = 'row_col_violation_positive_targets.txt'
VIOLATIONS_POSITIVE_TARGETS_FILENAME = 'violation_positive_targets.txt'
VIOLATIONS_POSITIVE_TRUTH_FILENAME = 'violation_positive_truth.txt'

SPLIT_IDS = ['train', 'test', 'valid']

PUZZLE_LABEL_POSITIVE = [1, 0]
PUZZLE_LABEL_NEGATIVE = [0, 1]

NEURAL_LEARNING_RATE = 0.001
NEURAL_LOSS = 'KLDivergence'
NEURAL_METRICS = ['categorical_accuracy']

# MNIST images are 28 x 28 = 784.
MNIST_DIMENSION = 28

def writeFile(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")

def readFile(path, dtype = None):
    rows = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if (line == ''):
                continue

            parts = line.split("\t")
            if (dtype is not None):
                parts = list(map(dtype, parts))

            rows.append(parts)

    return rows

def buildDigitNetwork(dimension):
    inputSize = MNIST_DIMENSION ** 2

    layers = [
        tensorflow.keras.layers.Input(shape=inputSize, name='input'),
        tensorflow.keras.layers.Reshape((MNIST_DIMENSION, MNIST_DIMENSION, 1), input_shape=(inputSize,), name='01-reshape'),
        tensorflow.keras.layers.Conv2D(filters=6, kernel_size=5, data_format='channels_last', name='03-conv2d_6_5'),
        tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_last', name='04-mp_2_2'),
        tensorflow.keras.layers.Activation('relu', name='05-relu'),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=5, data_format='channels_last', name='06-conv2d_16_5'),
        tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_last', name='07-mp_2_2'),
        tensorflow.keras.layers.Activation('relu', name='08-relu'),
        tensorflow.keras.layers.Flatten(name='09-flatten'),
        tensorflow.keras.layers.Dense(120, activation='relu', name='10-dense_120'),
        tensorflow.keras.layers.Dense(84, activation='relu', name='11-dense_84'),
        tensorflow.keras.layers.Dense(dimension, activation='softmax', name='output'),
    ]

    model = tensorflow.keras.Sequential(layers = layers, name = 'digitNetwork')

    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = NEURAL_LEARNING_RATE),
        loss = NEURAL_LOSS,
        metrics = NEURAL_METRICS
    )

    wrapper = pslpython.neupsl.NeuPSLWrapper(model, inputSize, dimension)
    # wrapper.model.summary()

    return wrapper

def writeData(outDir, dimension, rawPuzzleImages, rawPuzzleCellLabels, rawPuzzleLabels, labelMap = None):
    os.makedirs(outDir, exist_ok = True)

    # Map all labels to ints for PSL.
    # The first row of the first positive train puzzle will ALWAYS be [0, 1, 2, 3].
    # Other splits will use the label mapping from train.
    if (labelMap is None):
        firstPositivePuzzle = None
        labelMap = {}
        digitPinnedTruth = []

        for puzzleId in range(len(rawPuzzleLabels)):
            if (rawPuzzleLabels[puzzleId] != PUZZLE_LABEL_POSITIVE):
                continue

            firstPositivePuzzle = puzzleId

            for col in range(dimension):
                labelMap[rawPuzzleCellLabels[puzzleId][col]] = len(labelMap)

                for label in range(dimension):
                    digitPinnedTruth.append([puzzleId, 0, col, label, int(col == label)])

            break
        assert(firstPositivePuzzle is not None)

        if (dimension != len(labelMap)):
            raise ValueError("Cannot convert data that does not have (|labels| == dimension). |labels|: %d, dimension: %d. dir:' %s'." % (len(labelMap), dimension, outDir))

        writeFile(os.path.join(outDir, DIGIT_PINNED_TRUTH_FILENAME), digitPinnedTruth)
        writeFile(os.path.join(outDir, PUZZLE_POSITIVE_FIRST_ID_FILENAME), [[firstPositivePuzzle]])

    writeFile(os.path.join(outDir, DIGIT_LABELS_FILENAME), [[i] for i in range(dimension)])
    writeFile(os.path.join(outDir, LABEL_MAP_FILENAME), labelMap.items())

    # Write out two sets of data: all data and only positive puzzles.
    puzzleSets = [
        ['', lambda puzzleLabel: True],
        ['positive_', lambda puzzleLabel: puzzleLabel == PUZZLE_LABEL_POSITIVE],
    ]

    for (prefix, labelCheck) in puzzleSets:
        # Write out violation data.
        violations = []
        violationsTruth = []
        rowColViolations = []

        for puzzleId in range(len(rawPuzzleLabels)):
            if (not labelCheck(rawPuzzleLabels[puzzleId])):
                continue

            violations.append([puzzleId])
            violationsTruth.append([puzzleId, int(rawPuzzleLabels[puzzleId] != PUZZLE_LABEL_POSITIVE)])

            for row in range(dimension):
                for col in range(dimension):
                    rowColViolations.append([puzzleId, row, col])

        writeFile(os.path.join(outDir, prefix + VIOLATIONS_TARGETS_FILENAME), violations)
        writeFile(os.path.join(outDir, prefix + VIOLATIONS_TRUTH_FILENAME), violationsTruth)
        writeFile(os.path.join(outDir, prefix + ROW_VIOLATIONS_FILENAME), rowColViolations)

        digitFeatures = []

        for puzzleId in range(len(rawPuzzleLabels)):
            if (not labelCheck(rawPuzzleLabels[puzzleId])):
                continue

            for row in range(dimension):
                for col in range(dimension):
                    index = (row * dimension) + col
                    pixelStart = index * (MNIST_DIMENSION ** 2)
                    pixelEnd = (index + 1) * (MNIST_DIMENSION ** 2)

                    digitFeatures.append([puzzleId, row, col] + rawPuzzleImages[puzzleId][pixelStart:pixelEnd])

        writeFile(os.path.join(outDir, prefix + DIGIT_FEATURES_FILENAME), digitFeatures)

        digitTargets = []
        digitTruth = []

        for puzzleId in range(len(rawPuzzleLabels)):
            if (not labelCheck(rawPuzzleLabels[puzzleId])):
                continue

            for row in range(dimension):
                for col in range(dimension):
                    index = (row * dimension) + col
                    rawLabel = rawPuzzleCellLabels[puzzleId][index]

                    for label in range(dimension):
                        digitTargets.append([puzzleId, row, col, label])
                        digitTruth.append([puzzleId, row, col, label, int(labelMap[rawLabel] == label)])

        writeFile(os.path.join(outDir, prefix + DIGIT_TARGETS_FILENAME), digitTargets)
        writeFile(os.path.join(outDir, prefix + DIGIT_TRUTH_FILENAME), digitTruth)

    return labelMap

def readSourceData(sourceDir):
    puzzleImages = {}
    puzzleCellLabels = {}
    puzzleLabels = {}

    for splitId in SPLIT_IDS:
        puzzleImages[splitId] = readFile(os.path.join(sourceDir, splitId + '_' + PUZZLE_PIXELS_FILENAME), dtype = float)
        puzzleCellLabels[splitId] = readFile(os.path.join(sourceDir, splitId + '_' + CELL_LABELS_FILENAME))
        puzzleLabels[splitId] = readFile(os.path.join(sourceDir, splitId + '_' + PUZZLE_LABELS_FILENAME), dtype = int)

    return puzzleImages, puzzleCellLabels, puzzleLabels

def convertDir(sourceDir, baseSourceDir, baseOutDir, force):
    outDir = sourceDir.replace(baseSourceDir, baseOutDir)
    optionsPath = os.path.join(outDir, OPTIONS_FILENAME)
    if (os.path.isfile(optionsPath)):
        if (not force):
            print("Found existing PSL opions file, skipping generation. " + optionsPath)
            return

        print("Found existing PSL options file, but forcing over it. " + optionsPath)
        shutil.rmtree(outDir)

    with open(os.path.join(sourceDir, OPTIONS_FILENAME), 'r') as file:
        dataOptions = json.load(file)

    if (dataOptions['strategy'] in UNSUPPORTED_STRATEGIES):
        print('Strategy ' + dataOptions['strategy'] + ' not supported, skipping: ' + optionsPath)
        return

    print("Generating data defined in: " + optionsPath)
    os.makedirs(outDir, exist_ok = True)

    rawPuzzleImages, rawPuzzleCellLabels, rawPuzzleLabels = readSourceData(sourceDir)

    dimension = dataOptions['dimension']
    assert(dimension ** 2 == len(rawPuzzleCellLabels['train'][0]))

    modelWrapper = buildDigitNetwork(dimension)
    modelWrapper.save(
            os.path.join(outDir, UNTRAINED_DIGIT_MODEL_H5_FILENAME),
            os.path.join(outDir, UNTRAINED_DIGIT_MODEL_TF_DIRNAME))

    labelMap = writeData(os.path.join(outDir, 'learn'), dimension,
            rawPuzzleImages['train'], rawPuzzleCellLabels['train'], rawPuzzleLabels['train'])
    writeData(os.path.join(outDir, 'eval'), dimension,
            rawPuzzleImages['test'], rawPuzzleCellLabels['test'], rawPuzzleLabels['test'], labelMap)

    options = {
        'dataOptions': dataOptions,
        'timestamp': str(datetime.datetime.now()),
        'generator': os.path.basename(os.path.realpath(__file__)),
    }

    with open(optionsPath, 'w') as file:
        json.dump(options, file, indent = 4)

def fetchSourceDirs(baseDir):
    dirs = []

    for optionsPath in glob.glob(os.path.join(baseDir, '**', OPTIONS_FILENAME), recursive = True):
        dirs.append(os.path.dirname(optionsPath))

    return dirs

def main(arguments):
    source = os.path.abspath(os.path.realpath(arguments.source))
    outDir = os.path.abspath(os.path.realpath(arguments.outDir))

    sourceDirs = fetchSourceDirs(source)

    for sourceDir in sourceDirs:
        convertDir(sourceDir, source, outDir, arguments.force)

def _load_args():
    parser = argparse.ArgumentParser(description = 'Convert all VSPC data directories into PSL data directories.')

    parser.add_argument('--source', dest = 'source',
        action = 'store', type = str, default = DEFAULT_RAW_DATA_DIR,
        help = 'The existing base VSPC data directory (all subdirectories will be converted).')

    parser.add_argument('--force', dest = 'force',
        action = 'store_true', default = False,
        help = 'Ignore existing data directories and write over them.')

    parser.add_argument('--out-dir', dest = 'outDir',
        action = 'store', type = str, default = DEFAULT_DATA_DIR,
        help = 'Where to create PSL data directories.')

    return parser.parse_args()

if (__name__ == '__main__'):
    main(_load_args())
