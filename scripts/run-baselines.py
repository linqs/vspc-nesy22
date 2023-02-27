#!/usr/bin/env python3

# Run baselines after the data has already been created.

import argparse
import glob
import gzip
import json
import os
import re
import sys
import time

import numpy
import tensorflow

EXPERIMENT = 'vspc'

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(THIS_DIR, '..', 'results')
DEFAULT_RAW_DATA_DIR = os.path.join(THIS_DIR, '..', 'data', 'raw')
DATA_DIR = os.path.join(THIS_DIR, '..', 'data', EXPERIMENT)

OUT_FILENAME = 'out-baseline.json'
OUT_MODEL_DIRNAME = 'baseline-model-tf'
OUT_TRAIN_PREDICTIONS_FILENAME = 'baseline-train-predictions.txt.gz'
OUT_TRAIN_LABELS_FILENAME = 'baseline-train-labels.txt.gz'
OUT_TEST_PREDICTIONS_FILENAME = 'baseline-test-predictions.txt.gz'
OUT_TEST_LABELS_FILENAME = 'baseline-test-labels.txt.gz'

BASELINE_NAME_DIGIT = 'baseline-digit'
BASELINE_NAME_VISUAL = 'baseline-visual'
BASELINE_NAMES = [BASELINE_NAME_DIGIT, BASELINE_NAME_VISUAL]

TRAIN_LABELS_FILENAME = 'train_puzzle_labels.txt'
TEST_LABELS_FILENAME = 'test_puzzle_labels.txt'

# {method: (train, test)}
MODEL_FEATURES_FILENAME = {
    BASELINE_NAME_DIGIT: ('train_cell_labels.txt', 'test_cell_labels.txt'),
    BASELINE_NAME_VISUAL: ('train_puzzle_pixels.txt', 'test_puzzle_pixels.txt'),
}

MODEL_FEATURES_TYPE = {
    BASELINE_NAME_DIGIT: str,
    BASELINE_NAME_VISUAL: float,
}

OPTIONS_FILENAME = 'options.json'

EPOCHS = 100
NEURAL_LEARNING_RATE = 0.001

# MNIST images are 28 x 28 = 784.
MNIST_DIMENSION = 28

def writeFile(path, data, dtype = str, compress = False):
    if (compress):
        file = gzip.open(path, 'wt')
    else:
        file = open(path, 'w')

    for row in data:
        file.write('\t'.join([str(dtype(item)) for item in row]) + "\n")

    file.close()

def buildModel(method, dimension):
    assert(method in BASELINE_NAMES)

    if (method == BASELINE_NAME_DIGIT):
        inputSize = dimension ** 2

        layers = [
            tensorflow.keras.layers.Input(shape=inputSize, name='input'),
            tensorflow.keras.layers.Dense(512, activation='relu', name='01-dense_512'),
            tensorflow.keras.layers.Dense(512, activation='relu', name='02-dense_512'),
            tensorflow.keras.layers.Dense(256, activation='relu', name='03-dense_256'),
            tensorflow.keras.layers.Dense(2, activation='softmax', name='output'),
        ]
    else:
        inputSize = (dimension ** 2) * (MNIST_DIMENSION ** 2)
        visualPuzzleDim = MNIST_DIMENSION * dimension
        layers = [
            tensorflow.keras.layers.Input(shape=inputSize, name='input'),
            tensorflow.keras.layers.Reshape((visualPuzzleDim, visualPuzzleDim, 1), input_shape=(inputSize,), name='01-reshape'),
            tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='02-conv_16'),
            tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2), name='03-maxpool_2'),
            tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='04-conv_16'),
            tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2), name='05-maxpool_2'),
            tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='06-conv_16'),
            tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2), name='07-maxpool_2'),
            tensorflow.keras.layers.Flatten(name='08-flatten'),
            tensorflow.keras.layers.Dense(units=256, activation='relu', name='09-dense_256'),
            tensorflow.keras.layers.Dense(units=256, activation='relu', name='10-dense_256'),
            tensorflow.keras.layers.Dense(units=128, activation='relu', name='11-dense_128'),
            tensorflow.keras.layers.Dense(2, activation='softmax', name='output'),
        ]

    model = tensorflow.keras.Sequential(layers = layers, name = 'method')

    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = NEURAL_LEARNING_RATE),
        loss = 'binary_crossentropy',
        metrics = ['categorical_accuracy', tensorflow.keras.metrics.AUC()],
    )

    return model

def convertToInts(train, test):
    values = list(sorted(set(train.flatten().tolist()) | set(test.flatten().tolist())))

    valueMap = {}
    for value in values:
        valueMap[value] = len(valueMap)

    trainOut = []
    testOut = []

    for (outData, inData) in [[trainOut, train], [testOut, test]]:
        for row in inData:
            outData.append([valueMap[value] for value in row])

    train = numpy.stack(trainOut)
    test = numpy.stack(testOut)

    return train, test

def loadData(method, dataDir):
    trainLabels = numpy.loadtxt(os.path.join(dataDir, TRAIN_LABELS_FILENAME), delimiter = "\t", dtype = str)
    testLabels = numpy.loadtxt(os.path.join(dataDir, TEST_LABELS_FILENAME), delimiter = "\t", dtype = str)

    trainLabels, testLabels = convertToInts(trainLabels, testLabels)

    trainFilename, testFilename = MODEL_FEATURES_FILENAME[method]
    trainFeatures = numpy.loadtxt(os.path.join(dataDir, trainFilename), delimiter = "\t", dtype = MODEL_FEATURES_TYPE[method])
    testFeatures = numpy.loadtxt(os.path.join(dataDir, testFilename), delimiter = "\t", dtype = MODEL_FEATURES_TYPE[method])

    if (MODEL_FEATURES_TYPE[method] == str):
        trainFeatures, testFeatures = convertToInts(trainFeatures, testFeatures)

    return trainFeatures, trainLabels, testFeatures, testLabels

def runBaseline(method, baseDataDir, dataDir, baseOutDir, force):
    outDir = dataDir.replace(baseDataDir, os.path.join(baseOutDir, 'experiment::' + EXPERIMENT, 'method::' + method))
    optionsPath = os.path.join(outDir, OUT_FILENAME)

    if (os.path.isfile(optionsPath)):
        if (not force):
            print("Found existing results, skipping run. " + optionsPath)
            return

        print("Found existing results, but forcing over it. " + optionsPath)
        shutil.rmtree(outDir)

    with open(os.path.join(dataDir, OPTIONS_FILENAME), 'r') as file:
        dataOptions = json.load(file)

    assert('dimension' in dataOptions)

    print("Running run defined in: " + optionsPath)

    totalStartTime = int(time.time() * 1000)

    trainFeatures, trainLabels, testFeatures, testLabels = loadData(method, dataDir)

    model = buildModel(method, dataOptions['dimension'])
    # model.summary()

    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = NEURAL_LEARNING_RATE),
        loss = 'binary_crossentropy',
        metrics = ['categorical_accuracy', tensorflow.keras.metrics.AUC()],
    )

    startTime = int(time.time() * 1000)

    trainHistory = model.fit(trainFeatures, trainLabels, epochs = EPOCHS)

    endTime = int(time.time() * 1000)
    trainTime = endTime - startTime
    startTime = endTime

    loss, accuracy, auc = model.evaluate(testFeatures, testLabels)

    endTime = int(time.time() * 1000)
    testTime = endTime - startTime
    totalTime = endTime - totalStartTime

    trainPredictions = model.predict(trainFeatures)
    testPredictions = model.predict(testFeatures)

    print("%s Results -- Loss: %f, Accuracy: %f, AUROC: %f" % (method, loss, accuracy, auc))

    results = {
        'method': method,
        'epochs': EPOCHS,
        'neuralLearningRate': NEURAL_LEARNING_RATE,
        'loss': loss,
        'accuracy': accuracy,
        'AUROC': auc,
        'runtime': totalTime,
        'trainTime': trainTime,
        'testTime': testTime,
        'trainHistory': trainHistory.history,
    }

    os.makedirs(outDir, exist_ok = True)

    model.save(os.path.join(outDir, OUT_MODEL_DIRNAME), save_format = 'tf')

    writeFile(os.path.join(outDir, OUT_TRAIN_PREDICTIONS_FILENAME), trainPredictions, dtype = float, compress = True)
    writeFile(os.path.join(outDir, OUT_TRAIN_LABELS_FILENAME), trainLabels, dtype = int, compress = True)

    writeFile(os.path.join(outDir, OUT_TEST_PREDICTIONS_FILENAME), testPredictions, dtype = float, compress = True)
    writeFile(os.path.join(outDir, OUT_TEST_LABELS_FILENAME), testLabels, dtype = int, compress = True)

    with open(optionsPath, 'w') as file:
        json.dump(results, file, indent = 4)

def main(arguments):
    # Search for options files that indicate complete data dirs.
    for optionsPath in glob.glob(os.path.join(arguments.source, '**', OPTIONS_FILENAME), recursive = True):
        dataDir = os.path.dirname(optionsPath)

        for method in BASELINE_NAMES:
            runBaseline(method, arguments.source, dataDir, arguments.outDir, arguments.force)

def _load_args(args):
    parser = argparse.ArgumentParser(description = 'Run a CNN baseline for VSPC datasets.')

    parser.add_argument('--source', dest = 'source',
        action = 'store', type = str, default = DEFAULT_RAW_DATA_DIR,
        help = 'The existing base VSPC data directory (all subdirectories will be run).')

    parser.add_argument('--force', dest = 'force',
        action = 'store_true', default = False,
        help = 'Ignore existing results and write over them.')

    parser.add_argument('--out-dir', dest = 'outDir',
        action = 'store', type = str, default = DEFAULT_RESULTS_DIR,
        help = 'Where to output results.')

    return parser.parse_args()

if (__name__ == '__main__'):
    main(_load_args(sys.argv))
