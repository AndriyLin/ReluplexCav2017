/*********************                                                        */
/*! \file main.cpp
** \verbatim
** Top contributors (to current version):
**   Guy Katz
** This file is part of the Reluplex project.
** Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
** (in the top-level source directory) and their institutional affiliations.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**/

#include <cstdio>
#include <signal.h>

#include "AcasNeuralNetwork.h"
#include "File.h"
#include "Reluplex.h"
#include "MString.h"

//const char *FULL_NET_PATH = "./nnet/ACASXU_run2a_1_1_batch_2000.nnet";
const char *FULL_NET_PATH = "/Users/xuankang/Workspace/RepairML/tmp.nnet";

struct Index
{
    Index( unsigned newRow, unsigned newCol, unsigned newF )
        : row( newRow ), col( newCol ), f( newF )
    {
    }

    unsigned row;
    unsigned col;
    bool f;

    bool operator<( const Index &other ) const
    {
        if ( row != other.row )
            return row < other.row;
        if ( col != other.col )
            return col < other.col;

        if ( !f && other.f )
            return true;
        if ( f && !other.f )
            return false;

        return false;
    }
};

double normalizeInput( unsigned inputIndex, double value, AcasNeuralNetwork &neuralNetwork )
{
    double min = neuralNetwork._network->mins[inputIndex];
    double max = neuralNetwork._network->maxes[inputIndex];
    double mean = neuralNetwork._network->means[inputIndex];
    double range = neuralNetwork._network->ranges[inputIndex];

    if ( value < min )
        value = min;
    else if ( value > max )
        value = max;

    return ( value - mean ) / range;
}

double unnormalizeInput( unsigned inputIndex, double value, AcasNeuralNetwork &neuralNetwork )
{
    double mean = neuralNetwork._network->means[inputIndex];
    double range = neuralNetwork._network->ranges[inputIndex];

    return ( value * range ) + mean;
}

double unnormalizeOutput( double output, AcasNeuralNetwork &neuralNetwork )
{
    int inputSize = neuralNetwork._network->inputSize;
    double mean = neuralNetwork._network->means[inputSize];
    double range = neuralNetwork._network->ranges[inputSize];

    return ( output - mean ) / range;
}

double normalizeOutput( double output, AcasNeuralNetwork &neuralNetwork )
{
    int inputSize = neuralNetwork._network->inputSize;
    double mean = neuralNetwork._network->means[inputSize];
    double range = neuralNetwork._network->ranges[inputSize];

    return ( output * range ) + mean;
}

void getFixedInputs( Vector<double> &fixedInputs, unsigned pointToUse )
{
    /*
     * [0] 'support vector machine',
     * [1] 'bayesian',
     * [2] 'decision tree',
     * [3] 'conditional random field',
     *
     * [4] 'multilayer perceptron',
     * [5] 'convolutional',
     * [6] 'recurrent',
     * [7] 'sequence to sequence',
     *
     * [8] 'word embedding',
     * [9] 'generative adversarial'
     */
    // fixed cases are randomly generated by random_generator.py
    switch (pointToUse) {
        case 0:
            fixedInputs = {
                0.23078495873598626,  // support vector machine
                0.12856619768959154,  // bayesian
                0.8739096192491356,  // decision tree
                0.48785361779017633,  // conditional random field
                0.007546920915561195,  // multilayer perceptron
                0.2535721291837105,  // convolutional
                0.8790995909779602,  // recurrent
                0.14753102656391825,  // sequence to sequence
                0.27378291085218576,  // word embedding
                0.6593645648834594  // generative adversarial
            };
            break;

        case 1:
            fixedInputs = {
                0.5784150997991534,  // support vector machine
                0.7761657305157328,  // bayesian
                0.320561839718452,  // decision tree
                0.31720261821649554,  // conditional random field
                0.6184725152032041,  // multilayer perceptron
                0.7187815530598444,  // convolutional
                0.20393055292517548,  // recurrent
                0.5790541417997009,  // sequence to sequence
                0.06535743674991501,  // word embedding
                0.47680170703828006  // generative adversarial
            };
            break;

        case 2:
            fixedInputs = {
                0.6375268837973082,  // support vector machine
                0.2950167092961532,  // bayesian
                0.32806631431213984,  // decision tree
                0.20525656394241776,  // conditional random field
                0.43278762504430557,  // multilayer perceptron
                0.5018862713088322,  // convolutional
                0.009251637966326887,  // recurrent
                0.031542599159245066,  // sequence to sequence
                0.7579326613121927,  // word embedding
                0.668977670453346  // generative adversarial
            };
            break;

        case 3:
            fixedInputs = {
                0.19905362445677366,  // support vector machine
                0.26475937849068476,  // bayesian
                0.13729696226434618,  // decision tree
                0.8221227967720158,  // conditional random field
                0.5075341394978998,  // multilayer perceptron
                0.5444121840827625,  // convolutional
                0.08773122309746884,  // recurrent
                0.05613683662409075,  // sequence to sequence
                0.772730748906374,  // word embedding
                0.7283406298588497  // generative adversarial
            };
            break;

        default: {
            printf("Point-to-use out of bound.\n");
            exit(1);
            break;
        }
    }
    return;
}

void getFixedOutputs( const Vector<double> &fixedInputs,
                      Vector<double> &fixedOutputs,
                      unsigned outputLayerSize,
                      AcasNeuralNetwork &network )
{
    network.evaluate( fixedInputs, fixedOutputs, outputLayerSize );
}

void findMinimal( Vector<double> &fixedOutputs, unsigned &minimal )
{
    minimal = 0;

    for ( unsigned i = 1; i < fixedOutputs.size(); ++i )
        if ( fixedOutputs[i] < fixedOutputs[minimal] )
            minimal = i;
}

void findMaximal( Vector<double> &fixedOutputs, unsigned &maximal )
{
    maximal = 0;

    for ( unsigned i = 1; i < fixedOutputs.size(); ++i )
        if ( fixedOutputs[i] > fixedOutputs[maximal] )
            maximal = i;
}

Reluplex *lastReluplex = NULL;

void got_signal( int )
{
    printf( "Got signal\n" );

    if ( lastReluplex )
    {
        lastReluplex->quit();
    }
}

bool advMain( int argc, char **argv, int inputPoint, double inputDelta, unsigned runnerUp )
{
    String networkPath = FULL_NET_PATH;
    char *finalOutputFile;

    if ( argc < 3 )
        finalOutputFile = NULL;
    else
        finalOutputFile = argv[2];

    AcasNeuralNetwork neuralNetwork( networkPath.ascii() );

    unsigned numLayersInUse = neuralNetwork.getNumLayers() + 1;
    unsigned outputLayerSize = neuralNetwork.getLayerSize( numLayersInUse - 1 );

    printf( "Num layers in use: %u\n", numLayersInUse );
    printf( "Output layer size: %u\n", outputLayerSize );

    unsigned inputLayerSize = neuralNetwork.getLayerSize( 0 );

    unsigned numReluNodes = 0;
    for ( unsigned i = 1; i < numLayersInUse - 1; ++i )
        numReluNodes += neuralNetwork.getLayerSize( i );

    printf( "Input nodes = %u, relu nodes = %u, output nodes = %u\n", inputLayerSize, numReluNodes, outputLayerSize );

    Vector<double> fixedInputs;
    Vector<double> fixedOutputs;

    getFixedInputs( fixedInputs, inputPoint );
    getFixedOutputs( fixedInputs, fixedOutputs, outputLayerSize, neuralNetwork );

    unsigned maximal;
    findMaximal( fixedOutputs, maximal );

    if ( runnerUp == maximal ) {
        // same category, no need to compare
        return false;
    }

    printf( "Outputs are: \n" );
    for ( unsigned i = 0; i < fixedOutputs.size(); ++i )
    {
        printf( "\toutput[%u] = %lf\n", i, fixedOutputs[i] );
    }

    printf( "maximal: %u. runner up: %u\n", maximal, runnerUp );

    // Total size of the tableau:
    //   1. Input vars appear once
    //   2. Each internal var has a B instance, an F instance, and an auxiliary var for the B equation
    //   3. Each output var has an instance and an auxiliary var for its equation
    //   4. A single variable for the output constraints
    //   5. A single variable for the constants
    Reluplex reluplex( inputLayerSize + ( 3 * numReluNodes ) + ( 2 * outputLayerSize ) + 1 + 1,
                       finalOutputFile,
                       Stringf( "Point_%u_Delta_%.5lf_runnerUp_%u", inputPoint, inputDelta, runnerUp ) );

    lastReluplex = &reluplex;

    Map<Index, unsigned> nodeToVars;
    Map<Index, unsigned> nodeToAux;

    // We want to group variable IDs by layers.
    // The order is: f's from layer i, b's from layer i+1, aux variable for i+1, and repeat

    for ( unsigned i = 1; i < numLayersInUse; ++i )
    {
        unsigned currentLayerSize;
        if ( i + 1 == numLayersInUse )
            currentLayerSize = outputLayerSize;
        else
            currentLayerSize = neuralNetwork.getLayerSize( i );

        unsigned previousLayerSize = neuralNetwork.getLayerSize( i - 1 );

        // First add the f's from layer i-1
        for ( unsigned j = 0; j < previousLayerSize; ++j )
        {
            unsigned newIndex;

            newIndex = nodeToVars.size() + nodeToAux.size();
            nodeToVars[Index(i - 1, j, true)] = newIndex;
        }

        // Now add the b's from layer i
        for ( unsigned j = 0; j < currentLayerSize; ++j )
        {
            unsigned newIndex;

            newIndex = nodeToVars.size() + nodeToAux.size();
            nodeToVars[Index(i, j, false)] = newIndex;
        }

        // And now the aux variables from layer i
        for ( unsigned j = 0; j < currentLayerSize; ++j )
        {
            unsigned newIndex;

            newIndex = nodeToVars.size() + nodeToAux.size();
            nodeToAux[Index(i, j, false)] = newIndex;
        }
    }

    unsigned constantVar = nodeToVars.size() + nodeToAux.size();

    unsigned outputSlackVar = constantVar + 1;

    // Set bounds for constant var
    reluplex.setLowerBound( constantVar, 1.0 );
    reluplex.setUpperBound( constantVar, 1.0 );

    // Set bounds for inputs
    for ( unsigned i = 0; i < inputLayerSize ; ++i )
    {
        double realMax =
            ( neuralNetwork._network->maxes[i] - neuralNetwork._network->means[i] )
            / ( neuralNetwork._network->ranges[i] );
        double realMin =
            ( neuralNetwork._network->mins[i] - neuralNetwork._network->means[i] )
            / ( neuralNetwork._network->ranges[i] );

        double min = fixedInputs[i] - inputDelta;
        if ( min < realMin )
            min = realMin;

        double max = fixedInputs[i] + inputDelta;
        if ( max > realMax )
            max = realMax;

        printf( "Bounds for input %u: [ %.10lf, %.10lf ]\n", i, min, max );

        reluplex.setLowerBound( nodeToVars[Index(0, i, true)], min );
        reluplex.setUpperBound( nodeToVars[Index(0, i, true)], max );
    }

    // Set bounds for the output slack var. It's maximal - runnerUp,
    // so we want it to be negative - i.e., runner up scored higher.
    reluplex.setUpperBound( outputSlackVar, 0.0 );
    reluplex.markBasic( outputSlackVar );

    // Declare relu pairs and set bounds
    for ( unsigned i = 1; i < numLayersInUse - 1; ++i )
    {
        for ( unsigned j = 0; j < neuralNetwork.getLayerSize( i ); ++j )
        {
            unsigned b = nodeToVars[Index(i, j, false)];
            unsigned f = nodeToVars[Index(i, j, true)];

            reluplex.setReluPair( b, f );
            reluplex.setLowerBound( f, 0.0 );
        }
    }

    printf( "Number of auxiliary variables: %u\n", nodeToAux.size() );

    // Mark all aux variables as basic and set their bounds to zero
    for ( const auto &it : nodeToAux )
    {
        reluplex.markBasic( it.second );
        reluplex.setLowerBound( it.second, 0.0 );
        reluplex.setUpperBound( it.second, 0.0 );
    }

    // Populate the table
    for ( unsigned layer = 0; layer < numLayersInUse - 1; ++layer )
    {
        unsigned targetLayerSize;
        if ( layer + 2 == numLayersInUse )
            targetLayerSize = outputLayerSize;
        else
            targetLayerSize = neuralNetwork.getLayerSize( layer + 1 );

        for ( unsigned target = 0; target < targetLayerSize; ++target )
        {
            // This aux var will bind the F's from the previous layer to the B of this node.
            unsigned auxVar = nodeToAux[Index(layer + 1, target, false)];
            reluplex.initializeCell( auxVar, auxVar, -1 );

            unsigned bVar = nodeToVars[Index(layer + 1, target, false)];
            reluplex.initializeCell( auxVar, bVar, -1 );

            for ( unsigned source = 0; source < neuralNetwork.getLayerSize( layer ); ++source )
            {
                unsigned fVar = nodeToVars[Index(layer, source, true)];
                reluplex.initializeCell
                    ( auxVar,
                      fVar,
                      neuralNetwork.getWeight( layer, source, target ) );
            }

            // Add the bias via the constant var
            reluplex.initializeCell( auxVar,
                                     constantVar,
                                     neuralNetwork.getBias( layer + 1, target ) );
        }
    }

    // Slack var row: maximal - runnerUp (should be always > 0, so we look for maximal - runnerup < 0)
    unsigned maximalVar = nodeToVars[Index(numLayersInUse - 1, maximal, false)];
    unsigned runnerUpVar = nodeToVars[Index(numLayersInUse - 1, runnerUp, false)];
    reluplex.initializeCell( outputSlackVar, outputSlackVar, -1 );
    reluplex.initializeCell( outputSlackVar, maximalVar, 1 );
    reluplex.initializeCell( outputSlackVar, runnerUpVar, -1 );

    reluplex.setLogging( false );
    reluplex.setDumpStates( false );
    reluplex.toggleAlmostBrokenReluEliminiation( false );

    timeval start = Time::sampleMicro();
    timeval end;

    bool sat = false;

    try
    {
        Vector<double> inputs;
        Vector<double> outputs;

        double totalError = 0.0;

        Reluplex::FinalStatus result = reluplex.solve();
        if ( result == Reluplex::SAT )
        {
            printf( "Solution found!\n\n" );
            for ( unsigned i = 0; i < inputLayerSize; ++i )
            {
                double assignment = reluplex.getAssignment( nodeToVars[Index(0, i, true)] );
                printf( "input[%u] = %lf. Normalized: %lf.\n",
                        i, unnormalizeInput( i, assignment, neuralNetwork ), assignment );
                inputs.append( assignment );
            }

            printf( "\n" );
            for ( unsigned i = 0; i < outputLayerSize; ++i )
            {
                printf( "output[%u] = %.10lf. Normalized: %lf\n", i,
                        reluplex.getAssignment( nodeToVars[Index(numLayersInUse - 1, i, false)] ),
                        normalizeOutput( reluplex.getAssignment( nodeToVars[Index(numLayersInUse - 1, i, false)] ),
                                         neuralNetwork ) );
            }

            printf( "\nOutput using nnet.cpp:\n" );

            neuralNetwork.evaluate( inputs, outputs, outputLayerSize );
            unsigned i = 0;
            for ( const auto &output : outputs )
            {
                printf( "output[%u] = %.10lf. Normalized: %lf\n", i, output,
                        normalizeOutput( output, neuralNetwork ) );

                totalError +=
                    FloatUtils::abs( output -
                                     reluplex.getAssignment( nodeToVars[Index(numLayersInUse - 1, i, false)] ) );

                ++i;
            }

            printf( "\n" );
            printf( "Total error: %.10lf. Average: %.10lf\n", totalError, totalError / outputLayerSize );
            printf( "\nReminder: maximal: %u. runner up: %u\n", maximal, runnerUp );

            sat = true;
        }
        else if ( result == Reluplex::UNSAT )
        {
            printf( "Can't solve!\n" );
            sat = false;
        }
        else if ( result == Reluplex::ERROR )
        {
            printf( "Reluplex error!\n" );
        }
        else
        {
            printf( "Reluplex not done (quit called?)\n" );
        }

        printf( "Number of explored states: %u\n", reluplex.numStatesExplored() );
    }
    catch ( const Error &e )
    {
        printf( "main.cpp: Error caught. Code: %u. Errno: %i. Message: %s\n",
                e.code(),
                e.getErrno(),
                e.userMessage() );
        fflush( 0 );
    }

    end = Time::sampleMicro();

    unsigned milliPassed = Time::timePassed( start, end );
    unsigned seconds = milliPassed / 1000;
    unsigned minutes = seconds / 60;
    unsigned hours = minutes / 60;

    printf( "Total run time: %u milli (%02u:%02u:%02u)\n",
            Time::timePassed( start, end ), hours, minutes - ( hours * 60 ), seconds - ( minutes * 60 ) );

	return sat;
}

int main( int argc, char **argv )
{
    struct sigaction sa;
    memset( &sa, 0, sizeof(sa) );
    sa.sa_handler = got_signal;
    sigfillset( &sa.sa_mask );
    sigaction( SIGQUIT, &sa, NULL );

    if (argc < 2) {
        printf("Error! Need to specify which base point to check. Range [0, 3].\n");
        exit(1);
    }
    int point = atoi(argv[1]);

    List<double> deltas = { 0.05, 0.025, 0.01 };  // XL: 0.05 is large enough

    const int N_CATEGORY = 2;

    for ( const auto &delta : deltas )
    {
        bool sat = false;
        // XL: comparing all categories towards maximal
        unsigned i = 0;
        while ( ( !sat ) && ( i < N_CATEGORY ) )
        {
            printf( "Performing test for point %u, delta = %.5lf, part %u\n", point, delta, i + 1 );
            sat = advMain( argc, argv, point, delta, i );
            printf( "Test for point %u, delta = %.5lf, part %u DONE. Result = %s\n", point, delta, i + 1, sat ? "SAT" : "UNSAT" );
            printf( "\n\n" );
            ++i;
        }

        printf("---XL: Eventually, for delta %f, the result is %d.\n", delta, sat);
    }

    return 0;
}

//
// Local Variables:
// compile-command: "make -C .. "
// c-basic-offset: 4
// End:
//
