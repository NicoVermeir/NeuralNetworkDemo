using NeuralNetworkDemo.Helpers;
using NeuralNetworkDemo.Model;

namespace NeuralNetworkDemo.Chapters;

public static class Chapter3
{
    internal static void Run()
    {
        // Four nights out, and how each one should have gone
        (double[] Inputs, double Target)[] nights =
        [
          ([0, 0], 0),  // nobody: boring, stay home
          ([1, 0], 1),  // Alice: fun, go
          ([0, 1], 1),  // Bob: fun, go
          ([1, 1], 0),  // both: drama, stay home
        ];

        // The stranger from chapter 2
        var random = new Random(42);
        var network = new Network(
          Layer.CreateRandom(2, 2, random, Activation.Sigmoid),  // 2 in, 2 out
          Layer.CreateRandom(2, 1, random, Activation.Sigmoid)); // 2 in, 1 out

        for (int step = 0; step <= 2000; step++)
        {
            if (step % 200 == 0)
            {
                LossBar.Show(step, Training.Loss(network, nights));
            }

            foreach (var (inputs, target) in nights)
            {
                double said = network.Forward(inputs)[0];   // forward: decide
                double blame = said - target;         // regret: said minus should
                network.Backward([blame], learningRate: 1);  // backward: blame, nudge
            }
        }

        Party.Show((alice, bob) => network.Forward(alice, bob)[0]);
        foreach (var neuron in network.Layers.SelectMany(layer => layer.Neurons))
        {
            Console.WriteLine(neuron);
        }
        
        Plot.Show((alice, bob) => network.Forward(alice, bob)[0]);
        Console.WriteLine($"Both maybe: {network.Forward(0.5, 0.5)[0]:0.00}");
    }
}
