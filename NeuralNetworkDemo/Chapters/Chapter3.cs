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
          Layer.CreateRandom(2, 2, random, Activation.Step),  // 2 in, 2 out
          Layer.CreateRandom(2, 1, random, Activation.Step)); // 2 in, 1 out
    }
}
