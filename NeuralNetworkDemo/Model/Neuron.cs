using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public class Neuron(double[] weights, double bias, Activation activation)
{
    public double[] Weights { get; } = [.. weights];
    public double Bias { get; set;  } = bias;
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

    double[] lastInputs = [];  // remembered for Backward
    double lastSum;

    public double Forward(params double[] inputs)
    {
        if (inputs.Length != Weights.Length)
        {
            throw new ArgumentException($"Expected {Weights.Length} inputs but got {inputs.Length}");
        }
        lastInputs = inputs;
        lastSum = Dot(inputs, Weights) + Bias;
        return Activation.Apply(lastSum);
    }

    public double[] Backward(double blame, double learningRate)
    {
        double error = blame * Activation.Slope(lastSum);

        var inputBlame = new double[Weights.Length];
        for (int i = 0; i < Weights.Length; i++)
        {
            inputBlame[i] = error * Weights[i];  // the weight before the nudge
            Weights[i] -= learningRate * error * lastInputs[i];
        }
        Bias -= learningRate * error;
        return inputBlame;
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