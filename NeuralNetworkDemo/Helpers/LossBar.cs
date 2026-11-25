using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Helpers;

static class LossBar
{
    public static void Show(int step, double loss)
    {
        string bar = new('█', (int)Math.Round(Math.Min(loss, 0.6) * 100));
        Console.WriteLine($"{step,6}  {loss:0.000}  {bar}");
    }
}