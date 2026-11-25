using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public class Activation(string name, Func<double, double> apply)
{
    public static Activation Step = new ("Step", total => total >= 0 ? 1:0);

    public string Name { get; } = name;

    // What the neuron calls: its total in, its answer out
    public double Apply(double total) => apply(total);

    public override string ToString()
    {
        return Name;
    }
}
