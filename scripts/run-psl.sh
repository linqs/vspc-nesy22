#!/bin/bash

# Run all the hyperparameters.
# Requires the scripts/setup.sh script to be run first.

# An identifier to differentiate the output of this script/experiment from other scripts.
readonly EXPERIMENT_NAME='vspc'

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_OUT_DIR="${THIS_DIR}/../results"
readonly DATA_DIR="${THIS_DIR}/../data/${EXPERIMENT_NAME}"
readonly CLI_DIR="${THIS_DIR}/../cli"

readonly ADDITIONAL_PSL_OPTIONS='--postgres psl'

readonly SPLITS=$(seq -w 01 11)

function run_psl() {
    local outDir=$1
    local extraOptions=$2

    mkdir -p "${outDir}"

    local outPath="${outDir}/out.txt"
    local errPath="${outDir}/out.err"
    local timePath="${outDir}/time.txt"

    if [[ -e "${outPath}" ]]; then
        echo "Output file already exists, skipping: ${outPath}"
        return 0
    fi

    pushd . > /dev/null
        cd "${CLI_DIR}"

        # Run PSL.
        /usr/bin/time -v --output="${timePath}" ./run.sh ${extraOptions} > "${outPath}" 2> "${errPath}"

        # Copy any artifacts into the output directory.
        cp -r inferred-predicates "${outDir}/"
        cp *.data "${outDir}/"
        cp *.psl "${outDir}/"
    popd > /dev/null
}

function run() {
    local originalParamPath=$(grep "digit_model_untrained_tf" "${CLI_DIR}/visual-sudoku-learn.data" | sed "s#^.*data/${EXPERIMENT_NAME}/\(.*\)/digit_model_untrained_tf.*\$#\1#")

    # Splits are already specified in the data path, but do this loop so we complete full splits first.
    for split in ${SPLITS} ; do
        for optionsPath in $(find "${DATA_DIR}" -name options.json | grep "split::${split}" | sort) ; do
            # Skip large puzzles for now.
            if [[ "${optionsPath}" =~ dimension::9 ]] ; then
                continue
            fi

            local baseParamPath=$(dirname "${optionsPath}" | sed "s#^.*data/${EXPERIMENT_NAME}/##")
            local paramPath="${baseParamPath}"
            local options="${ADDITIONAL_PSL_OPTIONS}"

            # Change the .data files to use the current settings.
            sed -i "s#${originalParamPath}#${baseParamPath}#" "${CLI_DIR}/visual-sudoku-"{learn,eval}".data"

            local outDir="${BASE_OUT_DIR}/experiment::${EXPERIMENT_NAME}/method::neupsl/${paramPath}"

            echo "Running '${outDir}'."
            run_psl "${outDir}" "${options}"

            # Reset the .data files.
            sed -i "s#${baseParamPath}#${originalParamPath}#" "${CLI_DIR}/visual-sudoku-"{learn,eval}".data"
        done
    done
}

function main() {
    if [[ $# -ne 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    trap exit SIGINT

    run
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
