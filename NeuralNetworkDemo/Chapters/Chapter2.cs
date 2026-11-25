using NeuralNetworkDemo.Helpers;
using NeuralNetworkDemo.Model;

namespace NeuralNetworkDemo.Chapters;

public static class Chapter2
{
    internal static void Run()
    {
        // Fun? (at least one), No drama? (not both), then Go? (both say yes)
        var partyBrain = 
            new Network(
              new Layer(
                new Neuron([1, 1], bias: -0.5, Activation.Step),   // Fun?
                new Neuron([-1, -1], bias: 1.5, Activation.Step)),  // No drama?

              new Layer(
                new Neuron([1, 1], bias: -1.5, Activation.Step)));  // Go?

        Party.Show((alice, bob) => partyBrain.Forward(alice, bob)[0]);
        Plot.Show((alice, bob) => partyBrain.Forward(alice, bob)[0]);
    }
}
