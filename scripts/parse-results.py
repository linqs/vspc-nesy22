#!/usr/bin/env python3

# Parse out the results.

import glob
import json
import os
import re
import sys

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(THIS_DIR, '..', 'results')

LOG_FILENAME = 'out.txt'
BASELINE_FILENAME = 'out-baseline.json'

HEADER = [
    # Identifiers
    'method',
    'experiment',
    'dimension',
    'datasets',
    'strategy',
    'numTrain',
    'numTest',
    'numValid',
    'corruptChance',
    'overlap',
    'split',
    # Results
    'runtime',
    'digitAccuracy',
    'puzzleAccuracy',
    'puzzleAUROC',
]

# Parse a baseline results.
def parseBaseline(baselinePath):
    results = {}

    # Fetch the run identifiers off of the path.
    for (key, value) in re.findall(r'([\w\-\.,]+)::([\w\-\.,]+)', baselinePath):
        results[key] = value

    with open(baselinePath, 'r') as file:
        data = json.load(file)

    results['runtime'] = data['runtime']
    results['puzzleAccuracy'] = data['accuracy']
    results['puzzleAUROC'] = data['AUROC']

    return results

# Parse a PSL log for a NeuPSL run.
def parseLog(logPath):
    results = {}

    # Fetch the run identifiers off of the path.
    for (key, value) in re.findall(r'([\w\-\.,]+)::([\w\-\.,]+)', logPath):
        results[key] = value

    with open(logPath, 'r') as file:
        for line in file:
            line = line.strip()
            if (line == ''):
                continue

            match = re.search(r'^(\d+)\s+\[', line)
            if (match is not None):
                time = int(match.group(1))

            match = re.search(r'INFO  org.linqs.psl.application.inference.InferenceApplication  - Inference complete.', line)
            if (match is not None):
                results['runtime'] = time

            match = re.search(r'INFO  org.linqs.psl.cli.Launcher  - Evaluation results for PREDICTEDNUMBER -- Categorical Accuracy: ([\d\.,]+)', line)
            if (match is not None):
                results['digitAccuracy'] = float(match.group(1).replace(',', '.'))

            match = re.search(r'INFO  org.linqs.psl.cli.Launcher  - Evaluation results for VIOLATION -- Accuracy: ([\d\.,]+), ', line)
            if (match is not None):
                results['puzzleAccuracy'] = float(match.group(1).replace(',', '.'))

            match = re.search(r'INFO  org.linqs.psl.cli.Launcher  - Evaluation results for VIOLATION -- AUROC: ([\d\.,]+),', line)
            if (match is not None):
                results['puzzleAUROC'] = float(match.group(1).replace(',', '.'))

    # Check for an unfinished run.
    if ('runtime' not in results):
        return None

    return results

# [{key, value, ...}, ...]
def fetchResults():
    runs = []

    # Fetch PSL results.
    for logPath in glob.glob("%s/**/%s" % (RESULTS_DIR, LOG_FILENAME), recursive = True):
        run = parseLog(logPath)
        if (run is not None):
            runs.append(run)

    # Fetch baseline results.
    for baselinePath in glob.glob("%s/**/%s" % (RESULTS_DIR, BASELINE_FILENAME), recursive = True):
        run = parseBaseline(baselinePath)
        if (run is not None):
            runs.append(run)

    return runs

def main():
    runs = fetchResults()
    if (len(runs) == 0):
        return

    rows = []
    for run in runs:
        rows.append([run.get(key, '') for key in HEADER])

    print("\t".join(HEADER))
    for row in rows:
        print("\t".join(map(str, row)))

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s" % (executable), file = sys.stderr)
        sys.exit(1)

if (__name__ == '__main__'):
    _load_args(sys.argv)
    main()
