using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public static class Training
{
    // Regret: (said - should) squared, averaged over all nights
    public static double Loss(Network network, (double[] Inputs, double Target)[] examples)
    {
        double total = 0;

        foreach (var (inputs, target) in examples)
        {
            double error = network.Forward(inputs)[0] - target;
            total += error * error;
        }

        return total / examples.Length;
    }

    public static void Wiggle(Network network, (double[] Inputs, double Target)[] examples, double learningRate)
    {
        const double tiny = 0.0001;

        foreach (var layer in network.Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                for (int i = 0; i < neuron.Weights.Length; i++)
                {
                    double before = Loss(network, examples);
                    neuron.Weights[i] += tiny;

                    double slope = (Loss(network, examples) - before) / tiny;

                    neuron.Weights[i] -= tiny; //undo the nudge
                    neuron.Weights[i] -= learningRate * slope; // step downhill
                }

                double beforeBias = Loss(network, examples);
                neuron.Bias += tiny;

                double biasSlope = (Loss(network, examples) - beforeBias) / tiny;
                neuron.Bias -= tiny;
                neuron.Bias -= learningRate * biasSlope;
            }
        }
    }
}