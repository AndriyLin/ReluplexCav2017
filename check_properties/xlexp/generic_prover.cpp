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
** Addendum: some small modifications by Rudy Bunel, based on the
** property1/main.cpp file
** Copied from PLNN-Verification project.
**/

#include <cstdio>
#include <signal.h>

#include "AcasNeuralNetwork.h"
#include "File.h"
#include "Reluplex.h"
#include "MString.h"

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

Reluplex *lastReluplex = NULL;

void got_signal( int )
{
    printf( "Got signal\n" );
    if ( lastReluplex )
    {
        lastReluplex->quit();
    }
}

int main( int argc, char **argv )
{
    // This executable takes a path to a .nnet file as argument and
    // try to find a satisfiable assignment to the property
    // output =< 0
    struct sigaction sa;
    memset( &sa, 0, sizeof(sa) );
    sa.sa_handler = got_signal;
    sigfillset( &sa.sa_mask );
    sigaction( SIGQUIT, &sa, NULL );

    String networkPath;
    char *finalOutputFile;

    if ( argc < 2 )
    {
        printf( "Please specify a network\n" );
        exit( 1 );
    }
    else
        networkPath = argv[1];

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

    // Total size of the tableau:
    //   1. Input vars appear once
    //   2. Each internal var has a B instance, an F instance, and an auxiliary var for the B equation
    //   3. Each output var has an instance and an auxiliary var for its equation
    //   4. A single variable for the constants
    Reluplex reluplex( inputLayerSize + ( 3 * numReluNodes ) + ( 2 * outputLayerSize ) + 1,
                       finalOutputFile,
                       networkPath );

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

    // Set bounds for constant var
    reluplex.setLowerBound( constantVar, 1.0 );
    reluplex.setUpperBound( constantVar, 1.0 );

    // Set bounds for inputs
    for ( unsigned i = 0; i < inputLayerSize ; ++i )
    {
        double max =
            ( neuralNetwork._network->maxes[i] - neuralNetwork._network->means[i] )
            / ( neuralNetwork._network->ranges[i] );
        double min =
            ( neuralNetwork._network->mins[i] - neuralNetwork._network->means[i] )
            / ( neuralNetwork._network->ranges[i] );

        printf( "Bounds for input %u: [ %.10lf, %.10lf ]\n", i, min, max );

        reluplex.setLowerBound( nodeToVars[Index(0, i, true)], min );
        reluplex.setUpperBound( nodeToVars[Index(0, i, true)], max );
    }

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

    // The property in question: output[0] is lower or equal to 0
    int bound = 0;
    reluplex.setUpperBound( nodeToVars[Index(numLayersInUse - 1, 0, false)],
                            unnormalizeOutput( bound, neuralNetwork ) );

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

    reluplex.setLogging( false );
    reluplex.setDumpStates( false );
    reluplex.toggleAlmostBrokenReluEliminiation( false );

    timeval start = Time::sampleMicro();
    timeval end;

    try
    {
        Vector<double> inputs;
        Vector<double> outputs;

        double totalError = 0.0;

        printf( "\nTableau input ranges are:\n" );
        for ( unsigned i = 0; i < inputLayerSize ; ++i )
        {
            double min = reluplex.getLowerBound( nodeToVars[Index(0, i, true)] );
            double max = reluplex.getUpperBound( nodeToVars[Index(0, i, true)] );

            printf( "Bounds for input %u: [ %.2lf, %.2lf ]. Normalized: [ %.10lf, %.10lf ]\n",
                    i,
                    unnormalizeInput( i, min, neuralNetwork ),
                    unnormalizeInput( i, max, neuralNetwork ),
                    min,
                    max
                    );
        }
        printf( "\n\n" );

        reluplex.initialize();

        printf( "\nAfter tableau initialization, output ranges are:\n" );
        for ( unsigned i = 0; i < outputLayerSize ; ++i )
        {
            double max = reluplex.getUpperBound( nodeToVars[Index(numLayersInUse - 1, i, false)] );
            double min = reluplex.getLowerBound( nodeToVars[Index(numLayersInUse - 1, i, false)] );

            printf( "Bounds for output %u: [ %.10lf, %.10lf ]. Normalized: [ %.2lf, %.2lf ]\n",
                    i, min, max, normalizeOutput( min, neuralNetwork ), normalizeOutput( max, neuralNetwork ) );
        }
        printf( "\n\n" );

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

            printf( "\nOutput using nnet:\n" );

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
            printf( "\n" );
        }
        else if ( result == Reluplex::UNSAT )
        {
            printf( "Can't solve!\n" );
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

	return 0;
}

//
// Local Variables:
// compile-command: "make -C .. "
// tags-file-name: "../TAGS"
// c-basic-offset: 4
// End:
//
