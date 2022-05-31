#!/usr/bin/env python3

'''
Analyze the results.
The input to this script should be the output from parse-results.py, ex:
```
./scripts/parse-results.py > results.txt
./scripts/analyze-results.py AGGREGATE results.txt
```
'''

import argparse
import math
import os
import sqlite3
import sys

# Use this split if we have additional hyperparams.
VALID_SPLIT = 1

GROUP_COLS_STR = lambda prefix: ', '.join([prefix + '.' + col for col in GROUP_COLS])

METHOD_NAME_MAP = '''
    SELECT
        'baseline-digit' AS method,
        'Baseline Digit' AS name

    UNION ALL

    SELECT
        'baseline-visual' AS method,
        'Baseline Visual' AS name

    UNION ALL

    SELECT
        'neupsl' AS method,
        'NeuPSL' AS name
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

# Get all results with an actual value (e.g. ignore incomplete runs).
BASE_QUERY = '''
    SELECT *
    FROM Stats S
    WHERE puzzleAccuracy IS NOT NULL
    ORDER BY
        ''' + GROUP_COLS_STR('S') + '''
'''

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
    WHERE
        S.split != ''' + str(VALID_SPLIT) + '''
    GROUP BY
        ''' + GROUP_COLS_STR('S') + '''
    ORDER BY
        ''' + GROUP_COLS_STR('S') + '''
'''

GRAPH_SIMPLE_DATASETS = '''
    SELECT
        S.method,
        S.datasets,
        S.numTrain,
        S.numSplits,
        S.digitAccuracy_mean,
        S.digitAccuracy_std,
        S.puzzleAccuracy_mean,
        S.puzzleAccuracy_std,
        S.puzzleAUROC_mean,
        S.puzzleAUROC_std
    FROM
        (
            ''' + AGGREGATE_QUERY + '''
        ) S
    WHERE
        S.strategy = 'simple'
        AND S.dimension = 4
        AND S.overlap = 0.0
        AND numTrain <= 50
        AND numTest = 100
        AND numValid = 100
        AND corruptChance = 0.5
'''

TABLE_OVERLAP = '''
    SELECT
        S.*,
        S.[Number of Unique Images] / S.[Number of Images] AS 'Percentage Unique'
    FROM
        (
            SELECT
                UPPER(S.datasets) AS 'Data Source',
                M.name AS 'Model',
                S.numTrain AS 'Number of Puzzles',
                S.numTrain * 16 AS 'Number of Images',
                S.numTrain * 16 * (1.0 / (1.0 + S.overlap)) AS 'Number of Unique Images',
                S.numTrain * 16 * (1.0 / (1.0 + S.overlap)) / 4 AS 'Number of Unique Digit Instances',
                S.overlap,
                S.numSplits,
                S.puzzleAUROC_mean,
                S.puzzleAUROC_std
            FROM
                (
                    ''' + AGGREGATE_QUERY + '''
                ) S
                JOIN (
                    ''' + METHOD_NAME_MAP + '''
                ) M ON S.method = M.method
            WHERE
                S.strategy = 'simple'
                AND S.dimension = 4
                AND numTest = 100
                AND numValid = 100
                AND corruptChance = 0.5
        ) S
    ORDER BY
        S.[Data Source],
        S.[Number of Unique Images],
        S.Model,
        S.[Number of Puzzles]
'''

GRAPH_OVERLAP = '''
    SELECT
        O.*
    FROM
        (
            ''' + TABLE_OVERLAP + '''
        ) O
    WHERE
        O.[Number of Unique Images] IN (80, 160, 320)
    ORDER BY
        O.[Data Source],
        O.[Number of Unique Images],
        O.[Number of Puzzles],
        O.Model
'''

TABLE_SIMPLE_DATASETS = '''
    SELECT
        UPPER(S.datasets) AS 'Data Source',
        M.name AS 'Model',
        SUBSTR(CAST(ROUND(S.puzzleAUROC_mean, 2) AS TEXT) || '00', 0, 5)
            || ' ± '
            || SUBSTR(CAST(ROUND(S.puzzleAUROC_std, 2) AS TEXT) || '00', 0, 5)
            AS 'AuROC'
    FROM
        (
            ''' + AGGREGATE_QUERY + '''
        ) S
        JOIN (
            ''' + METHOD_NAME_MAP + '''
        ) M ON S.method = M.method
    WHERE
        S.strategy = 'simple'
        AND S.dimension = 4
        AND S.overlap = 0.0
        AND numTrain = 50
        AND numTest = 100
        AND numValid = 100
        AND corruptChance = 0.5
    ORDER BY
        S.datasets,
        S.method
'''

GRAPH_CROSS = '''
    SELECT
        UPPER(S.datasets) AS 'Data Source',
        M.name AS 'Model',
        s.numTrain,
        S.puzzleAUROC_mean,
        S.puzzleAUROC_std
    FROM
        (
            SELECT
                S.datasets,
                S.method,
                s.numTrain,
                S.puzzleAUROC_mean,
                S.puzzleAUROC_std
            FROM
                (
                    ''' + AGGREGATE_QUERY + '''
                ) S
            WHERE
                S.strategy = 'r_split'
                AND S.dimension = 4
                AND S.numTrain <= 50
                AND S.numTest = 100
                AND S.numValid = 100
                AND S.corruptChance = 0.5
                AND S.overlap = 0.0
        ) S
        JOIN (
            ''' + METHOD_NAME_MAP + '''
        ) M ON S.method = M.method
    ORDER BY
        S.datasets,
        S.method
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
    'GRAPH_SIMPLE_DATASETS': (
        GRAPH_SIMPLE_DATASETS,
        'Get the data for the simple datasets graph.',
    ),
    'GRAPH_OVERLAP': (
        GRAPH_OVERLAP,
        'Get the data for the overlap graph.',
    ),
    'GRAPH_CROSS': (
        GRAPH_CROSS,
        'Get the data for the cross graph.',
    ),
    'TABLE_SIMPLE_DATASETS': (
        TABLE_SIMPLE_DATASETS,
        'Get the results for the simple table.',
    ),
    'TABLE_OVERLAP': (
        TABLE_OVERLAP,
        'Get the data for the overlap table.',
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

def fetchQuery(mode, resultsPath):
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
    results = connection.execute(query)

    header = [column[0] for column in results.description]

    rows = []
    for row in results:
        rows.append(list(row))

    connection.close()

    return header, rows

def main(arguments):
    header, rows = fetchQuery(arguments.mode, arguments.results)

    if (arguments.latex):
        print(" & ".join(header) + ' \\\\')
        for row in rows:
            print(" & ".join(map(str, row)) + ' \\\\')

        return

    print("\t".join(header))
    for row in rows:
        print("\t".join(map(str, row)))

def _load_args():
    parser = argparse.ArgumentParser(description = 'Analyze already parsed results.')

    parser.add_argument('results',
        action = 'store', type = str,
        help = 'The path to the already parsed results.')

    parser.add_argument('mode',
        action = 'store', type = str,
        choices = list(sorted(RUN_MODES.keys())),
        help = 'The types of results to pull out.')

    parser.add_argument('--latex', dest = 'latex',
        action = 'store_true', default = False,
        help = 'Make output more compatible with LaTeX.')

    arguments = parser.parse_args()

    if (not os.path.isfile(arguments.results)):
        raise ValueError("Can't find the specified results path: " + arguments.results)

    return arguments

if (__name__ == '__main__'):
    main(_load_args())
