using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

public static class Letters
{
    // 27 symbols. The dot means "start" before a name and "end" after it.
    public const string Alphabet = ".abcdefghijklmnopqrstuvwxyz";

    public static int IndexOf(char letter) => Alphabet.IndexOf(letter);

    // "emm" -> 81 switches: three rows of 27, one switch on in each row
    public static double[] OneHot(string context)
    {
        var inputs = new double[context.Length * 27];
        for (int i = 0; i < context.Length; i++)
        {
            inputs[i * 27 + IndexOf(context[i])] = 1;
        }
        
        return inputs;
    }
}
