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

    IEnumerable<Neuron> AllNeurons => Layers.SelectMany(layer => layer.Neurons);

    public void Save(string path) =>
      File.WriteAllLines(path, AllNeurons.Select(neuron => neuron.Save()));

    public void Load(string path)
    {
        string[] lines = File.ReadAllLines(path);
        Neuron[] neurons = AllNeurons.ToArray();
        for (int i = 0; i < neurons.Length; i++)
        {
            neurons[i].Load(lines[i]);
        }
    }
}