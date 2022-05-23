#!/bin/bash

# Options can also be passed on the command line.
# These options are blind-passed to the CLI.
# Ex: ./run.sh -D log4j.threshold=DEBUG

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# TODO: Change when deploying.
# readonly PSL_VERSION='CANARY-2.3.1'
readonly PSL_VERSION='2.3.0-SNAPSHOT'
readonly JAR_PATH="./psl-neural-${PSL_VERSION}.jar"

readonly BASE_NAME='visual-sudoku'
readonly OUTPUT_DIRECTORY="${THIS_DIR}/inferred-predicates"

readonly ADDITIONAL_PSL_OPTIONS="\
    --int-ids \
    -D log4j.logger.org.deeplearning4j.nn=ERROR \
    -D log4j.logger.org.nd4j.linalg=ERROR \
    -D log4j.logger.org.nd4j.jita=ERROR \
    -D log4j.threshold=TRACE \
    -D modelpredicate.entityargs=0,1,2 \
    -D modelpredicate.labelargs=3 \
    -D modelpredicate.initialiterations=0 \
    -D modelpredicate.iterations=1 \
    -D admmreasoner.computeperiod=100 \
    -D admmreasoner.maxiterations=100 \
    -D admmreasoner.epsilonabs=1.0e-6 \
    -D admmreasoner.epsilonrel=1.0e-4 \
    -D reasoner.tolerance=1.0e-6 \
    -D inference.normalize=false \
    -D reasoner.simplexprojection=false \
"
readonly ADDITIONAL_WL_OPTIONS="--learn Energy \
    --supportingModels \
    --eval CategoricalEvaluator \
    -D weightlearning.evaluator=CategoricalEvaluator \
    -D categoricalevaluator.defaultpredicate=PredictedNumber \
    -D generalizedmargin.nonzeroregularization=100.0 \
    -D neural.tf.tensor.alpha.value=1.0 \
    -D weightlearning.fixnegativepriors=true \
    -D wla.gradientdescent.extension=EXPONENTIATED_GRADIENT \
    -D wla.gradientdescent.stepsize=1.0e-14 \
    -D wla.gradientdescent.numsteps=500 \
    -D wla.gradientdescent.runfulliterations=true \
    -D modelpredicate.batchsize=100000 \
"
readonly ADDITIONAL_EVAL_OPTIONS="\
    --infer ADMMInference \
    --eval CategoricalEvaluator \
    --eval DiscreteEvaluator \
    -D discreteevaluator.threshold=0.2 \
    --eval AUCEvaluator \
    -D aucevaluator.threshold=0.2 \
    -D inference.initialvalue=ZERO \
    -D admmreasoner.maxiterations=100 \
    -D modelpredicate.initialiterations=0 \
    -D modelpredicate.iterations=0 \
    -D modelpredicate.discretizeoutput=true \
"

function main() {
    trap exit SIGINT

    # Make sure we can run PSL.
    check_requirements
    fetch_psl

    # Run PSL.

    run_weight_learning "$@"

    run_inference "$@"
}

function run_weight_learning() {
    echo "Running PSL Weight Learning."

    java -jar "${JAR_PATH}" \
        --model "${THIS_DIR}/${BASE_NAME}.psl" \
        --data "${THIS_DIR}/${BASE_NAME}-learn.data" \
        ${ADDITIONAL_PSL_OPTIONS} ${ADDITIONAL_WL_OPTIONS} "$@"

    if [[ "$?" -ne 0 ]]; then
        echo 'ERROR: Failed to run weight learning.'
        exit 60
    fi
}

function run_inference() {
    echo "Running PSL Inference."

    java -jar "${JAR_PATH}" \
        --model "${THIS_DIR}/${BASE_NAME}-eval.psl" \
        --data "${THIS_DIR}/${BASE_NAME}-eval.data" \
        --output "${OUTPUT_DIRECTORY}" \
        ${ADDITIONAL_PSL_OPTIONS} ${ADDITIONAL_EVAL_OPTIONS} "$@"

    if [[ "$?" -ne 0 ]]; then
        echo 'ERROR: Failed to run infernce.'
        exit 70
    fi
}

function check_requirements() {
   local hasWget
   local hasCurl

   type wget > /dev/null 2> /dev/null
   hasWget=$?

   type curl > /dev/null 2> /dev/null
   hasCurl=$?

   if [[ "${hasWget}" -ne 0 ]] && [[ "${hasCurl}" -ne 0 ]]; then
      echo 'ERROR: wget or curl required to download the jar'
      exit 10
   fi

   type java > /dev/null 2> /dev/null
   if [[ "$?" -ne 0 ]]; then
      echo 'ERROR: java required to run project'
      exit 13
   fi
}

function get_fetch_command() {
   type curl > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "curl -o"
      return
   fi

   type wget > /dev/null 2> /dev/null
   if [[ "$?" -eq 0 ]]; then
      echo "wget -O"
      return
   fi

   echo 'ERROR: wget or curl not found'
   exit 20
}

function fetch_file() {
   local url=$1
   local path=$2
   local name=$3

   if [[ -e "${path}" ]]; then
      echo "${name} file found cached, skipping download."
      return
   fi

   echo "Downloading ${name} file located at: '${url}'."
   `get_fetch_command` "${path}" "${url}"
   if [[ "$?" -ne 0 ]]; then
      echo "ERROR: Failed to download ${name} file"
      exit 30
   fi
}

# Fetch the jar from a remote or local location and put it in this directory.
# Snapshots are fetched from the local maven repo and other builds are fetched remotely.
function fetch_psl() {
   if [[ $PSL_VERSION == *'SNAPSHOT'* ]]; then
      local snapshotJARPath="$HOME/.m2/repository/org/linqs/psl-neural/${PSL_VERSION}/psl-neural-${PSL_VERSION}.jar"
      cp "${snapshotJARPath}" "${JAR_PATH}"
   else
      local remoteJARURL="https://repo1.maven.org/maven2/org/linqs/psl-neural/${PSL_VERSION}/psl-neural-${PSL_VERSION}.jar"
      fetch_file "${remoteJARURL}" "${JAR_PATH}" 'psl-jar'
   fi
}

main "$@"
