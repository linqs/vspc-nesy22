#!/usr/bin/env python3

'''
Analyze the results.
The input to this script should be the output from parse-results.py, ex:
```
./scripts/parse-results.py > results.txt
./scripts/analyze-results.py AGGREGATE results.txt
```
'''

import math
import os
import sqlite3
import sys

# Get all results with an actual value (e.g. ignore incomplete runs).
BASE_QUERY = '''
    SELECT *
    FROM Stats
    WHERE puzzleAccuracy IS NOT NULL
'''

GROUP_COLS = [
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
]

GROUP_COLS_STR = lambda prefix: ', '.join([prefix + '.' + col for col in GROUP_COLS])

# Aggregate over splits and iterations.
AGGREGATE_QUERY = '''
    SELECT
        ''' + GROUP_COLS_STR('S') + ''',
        COUNT(S.split) AS numSplits,
        AVG(S.digitAccuracy) AS digitAccuracy_mean,
        STDEV(S.digitAccuracy) AS digitAccuracy_std,
        AVG(S.puzzleAccuracy) AS puzzleAccuracy_mean,
        STDEV(S.puzzleAccuracy) AS puzzleAccuracy_std,
        AVG(S.puzzleAUROC) AS puzzleAUROC_mean,
        STDEV(S.puzzleAUROC) AS puzzleAUROC_std
    FROM
        (
            ''' + BASE_QUERY + '''
        ) S
    GROUP BY
        ''' + GROUP_COLS_STR('S') + '''
    ORDER BY
        ''' + GROUP_COLS_STR('S') + '''
'''

# Get the best set of hyperparams for each data setting.
# "Best" is determined by best puzzle AUROC.
BEST_HYPERPARAMS = '''
    SELECT
        ''' + GROUP_COLS_STR('S') + '''
    FROM (
        SELECT
            S.*,
            ROW_NUMBER() OVER RankWindow AS rank
        FROM Stats S
        WHERE S.split = 0
        WINDOW RankWindow AS (
            PARTITION BY
                ''' + GROUP_COLS_STR('S') + '''
            ORDER BY
                S.puzzleAUROC,
                S.puzzleAccuracy
        )
    ) S
    WHERE S.rank = 1
    ORDER BY
        ''' + GROUP_COLS_STR('S') + '''
'''

BEST_RUNS = '''
    SELECT
        ''' + GROUP_COLS_STR('S') + ''',
        COUNT(S.split) AS numSplits,
        AVG(S.digitAccuracy) AS digitAccuracy_mean,
        STDEV(S.digitAccuracy) AS digitAccuracy_std,
        AVG(S.puzzleAccuracy) AS puzzleAccuracy_mean,
        STDEV(S.puzzleAccuracy) AS puzzleAccuracy_std,
        AVG(S.puzzleAUROC) AS puzzleAUROC_mean,
        STDEV(S.puzzleAUROC) AS puzzleAUROC_std
    FROM
        Stats S
        JOIN (
            ''' + BEST_HYPERPARAMS + '''
        ) H ON
            S.method = H.method
            AND S.experiment = H.experiment
            AND S.dimension = H.dimension
            AND S.datasets = H.datasets
            AND S.strategy = H.strategy
            AND S.numTrain = H.numTrain
            AND S.numTest = H.numTest
            AND S.numValid = H.numValid
            AND S.corruptChance = H.corruptChance
            AND S.overlap = H.overlap
    WHERE S.split != 0
    GROUP BY
        ''' + GROUP_COLS_STR('S') + '''
    ORDER BY
        ''' + GROUP_COLS_STR('S') + '''
'''

BOOL_COLUMNS = {
}

INT_COLUMNS = {
    'dimension',
    'numTrain',
    'numTest',
    'numValid',
    'split',
    'runtime',
}

FLOAT_COLUMNS = {
    'corruptChance',
    'overlap',
    'digitAccuracy',
    'puzzleAccuracy',
    'puzzleAUROC',
}

# {key: (query, description), ...}
RUN_MODES = {
    'BASE': (
        BASE_QUERY,
        'Just get the results with no additional processing.',
    ),
    'AGGREGATE': (
        AGGREGATE_QUERY,
        'Aggregate over split.',
    ),
    'BEST_HYPERPARAMS': (
        BEST_HYPERPARAMS,
        'Get the best hyperparams for each data setting.',
    ),
    'BEST_RUNS': (
        BEST_RUNS,
        'Get the best non-validation runs.',
    ),
}

# ([header, ...], [[value, ...], ...])
def fetchResults(path):
    rows = []
    header = None

    with open(path, 'r') as file:
        for line in file:
            line = line.strip("\n ")
            if (line == ''):
                continue

            row = line.split("\t")

            # Get the header first.
            if (header is None):
                header = row
                continue

            assert(len(header) == len(row))

            for i in range(len(row)):
                if (row[i] == ''):
                    row[i] = None
                elif (header[i] in BOOL_COLUMNS):
                    row[i] = (row[i].upper() == 'TRUE')
                elif (header[i] in INT_COLUMNS):
                    row[i] = int(row[i])
                elif (header[i] in FLOAT_COLUMNS):
                    row[i] = float(row[i])

            rows.append(row)

    return header, rows

# Standard deviation UDF for sqlite3.
# Taken from: https://www.alexforencich.com/wiki/en/scripts/python/stdev
class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k-2))

def main(mode, resultsPath):
    columns, data = fetchResults(resultsPath)
    if (len(data) == 0):
        return

    quotedColumns = ["'%s'" % column for column in columns]

    columnDefs = []
    for i in range(len(columns)):
        column = columns[i]
        quotedColumn = quotedColumns[i]

        if (column in BOOL_COLUMNS):
            columnDefs.append("%s INTEGER" % (quotedColumn))
        elif (column in INT_COLUMNS):
            columnDefs.append("%s INTEGER" % (quotedColumn))
        elif (column in FLOAT_COLUMNS):
            columnDefs.append("%s FLOAT" % (quotedColumn))
        else:
            columnDefs.append("%s TEXT" % (quotedColumn))

    connection = sqlite3.connect(":memory:")
    connection.create_aggregate("STDEV", 1, StdevFunc)

    connection.execute("CREATE TABLE Stats(%s)" % (', '.join(columnDefs)))

    connection.executemany("INSERT INTO Stats(%s) VALUES (%s)" % (', '.join(columns), ', '.join(['?'] * len(columns))), data)

    query = RUN_MODES[mode][0]
    rows = connection.execute(query)

    print("\t".join([column[0] for column in rows.description]))
    for row in rows:
        print("\t".join(map(str, row)))

    connection.close()

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 2 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <results path> <mode>" % (executable), file = sys.stderr)
        print("modes:", file = sys.stderr)
        for (key, (query, description)) in RUN_MODES.items():
            print("    %s - %s" % (key, description), file = sys.stderr)
        sys.exit(1)

    resultsPath = args.pop(0)
    if (not os.path.isfile(resultsPath)):
        raise ValueError("Can't find the specified results path: " + resultsPath)

    mode = args.pop(0).upper()
    if (mode not in RUN_MODES):
        raise ValueError("Unknown mode: '%s'." % (mode))

    return mode, resultsPath

if (__name__ == '__main__'):
    main(*_load_args(sys.argv))
