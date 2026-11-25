using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public class Activation(string name, Func<double, double> apply, Func<double, double> slope)
{
    public static Activation Step = new ("Step", total => total >= 0 ? 1:0, _ => 0);

    public static Activation Sigmoid = new ("Sigmoid", Sig, total => Sig(total) * (1 - Sig(total)));

    public static Activation Identity = new ("Identity", total => total, total => 1);


    static double Sig(double total) => 1 / (1 + Math.Exp(-total));

    public string Name { get; } = name;
    public Func<double, double> Slope { get; } = slope;

    // What the neuron calls: its total in, its answer out
    public double Apply(double total) => apply(total);

    public override string ToString()
    {
        return Name;
    }
}
