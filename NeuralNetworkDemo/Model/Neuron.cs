using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public class Neuron(double[] weights, double bias, Activation activation)
{
    public double[] Weights { get; } = [.. weights];
    public double Bias { get; } = bias;
    public Activation Activation { get; } = activation;

    // Chapter 2: every network starts as random numbers
    public static Neuron CreateRandom(int inputCount, Random random, Activation activation)
    {
        double limit = 1 / Math.Sqrt(inputCount);
        double Next() => (random.NextDouble() * 2 - 1) * limit;  // -limit..+limit

        var weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++)
        {
            weights[i] = Next();
        }

        return new Neuron(weights, Next(), activation);
    }

    public double Forward(params double[] inputs)
    {
        if (inputs.Length != weights.Length)
        {
            throw new ArgumentException($"Expected {weights.Length} inputs but got {inputs.Length}");
        }

        return Activation.Apply(Dot(inputs, weights) + Bias);
    }

    public double Dot(double[] a, double[] b)
    {
        double sum = 0;

        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    public override string ToString()
    {
        var weights = string.Join(", ", Weights.Select(w => w.ToString("0.00")));
        return $"{Activation}: weights [{weights}], bias {Bias:0.00}";
    }
}