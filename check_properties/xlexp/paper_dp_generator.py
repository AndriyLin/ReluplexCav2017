#! /usr/bin/env python3
""" Randomly generate data points for Paper experiment. """

import random

def generate(k):
    texts = [
        'support vector machine',
        'bayesian',
        'decision tree',
        'conditional random field',

        'multilayer perceptron',
        'convolutional',
        'recurrent',
        'sequence to sequence',

        'word embedding',
        'generative adversarial'
    ]

    ds = [random.uniform(0, 1) for _ in range(k)]
    ds = [str(v) for v in ds]
    final = []
    for i, s in enumerate(ds):
        if i < len(ds) - 1:
            final.append(s + ',  // ' + texts[i] + '\n')
        else:
            final.append(s + '  // ' + texts[i] + '\n')
        pass
    for s in final:
        print(s, end='')
        pass
    return


if __name__ == '__main__':
    generate(10)
    pass
