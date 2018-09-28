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
    //16 ['Amy', 'Ashley', 'Emily', 'Emma', 'Isabella', 'Jennifer', 'Jessica', 'Linda', 'Mary', 'Melissa', 'Michelle', 'Olivia', 'Patricia', 'Sarah', 'Sophia', 'Susan']
    switch (pointToUse) {
        default: {
            // 0.01 * 14 + 0.36 * 2
            fixedInputs = {
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.43,  // [7] Linda
                    0.43,  // [8] Mary
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01
            };
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

bool advMain( int argc, char **argv, unsigned inputPoint, double inputDelta, unsigned runnerUp, bool onlyMaryLinda )
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
        // same category, no need it compare
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
    //   6. XL: one more variable for input sum-to-1.0 constraint
    //   7. XL: one more variable for Mary+Linda=fixed constraint
    Reluplex reluplex( inputLayerSize + ( 3 * numReluNodes ) + ( 2 * outputLayerSize ) + 1 + 1
                        + 1 + 1,
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

    unsigned inputConstraintVar = outputSlackVar + 1;
    unsigned maryLindaConstraintVar = inputConstraintVar + 1;

    // Set bounds for constant var
    reluplex.setLowerBound( constantVar, 1.0 );
    reluplex.setUpperBound( constantVar, 1.0 );

    unsigned idx_mary = 8;  // in size 16 for current experiment
    unsigned idx_linda = 7;  // in size 16 for current experiment

    // Set bounds for inputs
    for ( unsigned i = 0; i < inputLayerSize ; ++i )
    {
        double realMax =
            ( neuralNetwork._network->maxes[i] - neuralNetwork._network->means[i] )
            / ( neuralNetwork._network->ranges[i] );
        double realMin =
            ( neuralNetwork._network->mins[i] - neuralNetwork._network->means[i] )
            / ( neuralNetwork._network->ranges[i] );

        double min = fixedInputs[i];
        double max = fixedInputs[i];

        if (!onlyMaryLinda || i == idx_mary || i == idx_linda) {
            min = fixedInputs[i] - inputDelta;
            if ( min < realMin )
                min = realMin;

            max = fixedInputs[i] + inputDelta;
            if ( max > realMax )
                max = realMax;
        }

        printf( "Bounds for input %u: [ %.10lf, %.10lf ]\n", i, min, max );

        reluplex.setLowerBound( nodeToVars[Index(0, i, true)], min );
        reluplex.setUpperBound( nodeToVars[Index(0, i, true)], max );
    }

    // Set bounds for the output slack var. It's maximal - runnerUp,
    // so we want it to be negative - i.e., runner up scored higher.
    reluplex.setUpperBound( outputSlackVar, 0.0 );
    reluplex.markBasic( outputSlackVar );

    // XL: the input elements should sum to 1.0, exactly 1.0
    reluplex.markBasic( inputConstraintVar );
    reluplex.setLowerBound( inputConstraintVar, 1.0 );
    reluplex.setUpperBound( inputConstraintVar, 1.0 );

    // XL: Mary and Linda should sum to original sum, exactly original sum. But their individual portion could change.
    reluplex.markBasic( maryLindaConstraintVar );
    reluplex.setLowerBound( maryLindaConstraintVar, fixedInputs[idx_mary] + fixedInputs[idx_linda]);
    reluplex.setUpperBound( maryLindaConstraintVar, fixedInputs[idx_mary] + fixedInputs[idx_linda] );

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

    // This is the input constraint "all inputs sum to 1.0".
    reluplex.initializeCell( inputConstraintVar, inputConstraintVar, -1.0 );
    for ( unsigned i = 0; i < inputLayerSize; ++i ) {
        unsigned inputTargetVar = nodeToVars[Index(0, i, true)];
        reluplex.initializeCell( inputConstraintVar, inputTargetVar, 1.0 );
    }

    // This is the constraint that "Mary + Linda = original sum".
    reluplex.initializeCell( maryLindaConstraintVar, maryLindaConstraintVar, -1.0 );
    reluplex.initializeCell( maryLindaConstraintVar, nodeToVars[Index(0, idx_mary, true)], 1.0 );
    reluplex.initializeCell( maryLindaConstraintVar, nodeToVars[Index(0, idx_linda, true)], 1.0 );

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
    if ( argc < 2 )
    {
        printf( "Error! Please specify if only Mary/Linda percentage will be perturbed. 1/0 for yes/no.\n" );
        exit( 1 );
    }
    bool onlyMaryLinda = bool(atoi( argv[1] ));

    struct sigaction sa;
    memset( &sa, 0, sizeof(sa) );
    sa.sa_handler = got_signal;
    sigfillset( &sa.sa_mask );
    sigaction( SIGQUIT, &sa, NULL );

    //List<unsigned> points = { 0, 1, 2, 3, 4 };
    List<unsigned> points = { 0 };  // XL: just test one point for demo
    List<double> deltas = { 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01 };  // XL: try 0.3, 0.2 as well

    for ( const auto &point : points )
    {
        for ( const auto &delta : deltas )
        {
            bool sat = false;
            // XL: comparing [0, 1, 2, 3] against maximal
            unsigned i = 0;
            while ( ( !sat ) && ( i < 4 ) )
            {
                printf( "Performing test for point %u, delta = %.5lf, part %u\n", point, delta, i + 1 );
                sat = advMain( argc, argv, point, delta, i, onlyMaryLinda );
                printf( "Test for point %u, delta = %.5lf, part %u DONE. Result = %s\n", point, delta, i + 1, sat ? "SAT" : "UNSAT" );
                printf( "\n\n" );
                ++i;
            }

            printf("---XL: Eventually, for delta %f, the result is %d.\n", delta, sat);
        }
    }

    return 0;
}

//
// Local Variables:
// compile-command: "make -C .. "
// c-basic-offset: 4
// End:
//
