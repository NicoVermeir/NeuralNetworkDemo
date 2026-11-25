using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public class Layer(params Neuron[] neurons)
{
    public Neuron[] Neurons { get; } = neurons;

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