using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Helpers;

static class Plot
{
    // An even size keeps grid points off the 0.5 and 1.5 lines: clean edges
    public static void Show(Func<double, double, double> brain, int size = 16)
    {
        for (int row = size - 1; row >= 0; row--)
        {
            for (int col = 0; col < size; col++)
            {
                double answer = brain(col / (size - 1.0), row / (size - 1.0));
                Console.Write(answer >= 0.5 ? "██" : "··");
            }
            Console.WriteLine();
        }
    }
}