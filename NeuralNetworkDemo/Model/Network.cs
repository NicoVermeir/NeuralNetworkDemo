using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public class Network(params Layer[] layers)
{
    public Layer[] Layers { get; } = layers;

    public double[] Forward(params double[] inputs)
    {
        var values = inputs;

        foreach (var layer in Layers)
        {
            values = layer.Forward(values);
        }

        return values;
    }

    public void Backward(double[] blame, double learningRate)
    {
        for (int i = Layers.Length - 1; i >= 0; i--)
        {
            blame = Layers[i].Backward(blame, learningRate);
        }
    }
}