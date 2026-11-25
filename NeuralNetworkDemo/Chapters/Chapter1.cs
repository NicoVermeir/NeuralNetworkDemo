using NeuralNetworkDemo.Model;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Chapters;

public static class Chapter1
{
    internal static void Run()
    {
        // The homebody: Alice counts 1, Bob counts 1, and the mood is -1.5
        var homebody = new Neuron([1, 1], bias: -1.5, Activation.Step);

        Party.Show((alice, bob) => homebody.Forward(alice, bob));
        Console.WriteLine(homebody);
    }
}
