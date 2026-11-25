using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public class Layer(params Neuron[] neurons)
{
    public Neuron[] Neurons { get; } = neurons;

    public static Layer CreateRandom(int inputCount, int neuronCount, Random random, Activation activation)
    {
        var neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
        {
            neurons[i] = Neuron.CreateRandom(inputCount, random, activation);
        }

        return new Layer(neurons);
    }

    public double[] Forward(double[] inputs)
    {
        var outputs = new double[Neurons.Length];

        for (int i = 0; i < Neurons.Length; i++)
        {
            outputs[i] = Neurons[i].Forward(inputs);
        }

        return outputs;
    }
}