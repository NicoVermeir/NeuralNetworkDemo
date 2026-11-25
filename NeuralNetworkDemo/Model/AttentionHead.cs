using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkDemo.Model;

// One attention head: every word asks a question, wears a name tag,
// and has something to pass on
public class AttentionHead(Layer query, Layer key, Layer value)
{
    public Layer Query { get; } = query;  // What am I looking for?
    public Layer Key { get; } = key;    // What's on my name tag?
    public Layer Value { get; } = value;  // What do I pass on?

    public (double[][] Outputs, double[][] Attention) Forward(double[][] words)
    {
        var query = words.Select(Query.Forward).ToArray();
        var key = words.Select(Key.Forward).ToArray();
        var value = words.Select(Value.Forward).ToArray();

        var outputs = new double[words.Length][];
        var attention = new double[words.Length][];

        for (int i = 0; i < words.Length; i++)
        {
            var scores = new double[i + 1];
            for (int j = 0; j <= i; j++)  // look back, never ahead
            {
                scores[j] = Neuron.Dot(query[i], key[j]) / Math.Sqrt(key[j].Length);
            }

            attention[i] = Neuron.Softmax(scores);

            outputs[i] = new double[value[i].Length];
            for (int j = 0; j <= i; j++)
            {
                for (int d = 0; d < outputs[i].Length; d++)
                {
                    outputs[i][d] += attention[i][j] * value[j][d];
                }
            }

        }

        return (outputs, attention);
    }
}