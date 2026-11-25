using Microsoft.ML.Tokenizers;
using NeuralNetworkDemo.Model;

namespace NeuralNetworkDemo.Chapters;

public static class Chapter5
{
    internal static void Run()
    {
        var tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");
        string[] texts = ["Alice told Bob she misses him"];
        // Embeddings, made by hand: [name, female, male, pronoun]
        var embeddings = new Dictionary<string, double[]>
        {
            ["Alice"] = [1, 1, 0, 0],
            ["Bob"] = [1, 0, 1, 0],
            ["she"] = [0, 1, 0, 1],
            ["her"] = [0, 1, 0, 1],
            ["he"] = [0, 0, 1, 1],
            ["him"] = [0, 0, 1, 1],
            ["told"] = [0, 0, 0, 0],
            ["misses"] = [0, 0, 0, 0],
        };

        // Query, Key and Value: three Layers of Identity neurons, wired by hand
        Neuron Wire(params double[] weights) => new(weights, 0, Activation.Identity);

        var head = new AttentionHead(
          // I look for: a name (if I'm a pronoun), female, male. Strength 5.
          query: new Layer(Wire(0, 0, 0, 5), Wire(0, 5, 0, 0), Wire(0, 0, 5, 0)),
          // My name tag: name, female, male
          key: new Layer(Wire(1, 0, 0, 0), Wire(0, 1, 0, 0), Wire(0, 0, 1, 0)),
          // What I pass on: all of me
          value: new Layer(Wire(1, 0, 0, 0), Wire(0, 1, 0, 0),
                   Wire(0, 0, 1, 0), Wire(0, 0, 0, 1)));

        foreach (string text in texts)
        {
            var tokens = tokenizer.EncodeToTokens(text, out _);
            Console.WriteLine($"{text}: {tokens.Count} tokens");
            foreach (var token in tokens)
            {
                Console.WriteLine($" {token.Id,7} '{token.Value}'");
            }
        }

        Attend("Alice told Bob he misses her");

        // Run the head on a sentence and print who looks at whom
        void Attend(string sentence)
        {
            string[] words = sentence.Split(' ');
            double[][] input = words.Select(word => embeddings[word]).ToArray();
            var (outputs, attention) = head.Forward(input);

            Console.WriteLine(); 
            Console.Write(new string(' ', 8));
            foreach (string word in words) Console.Write(word.PadLeft(8));
            Console.WriteLine();
            for (int i = 0; i < words.Length; i++)
            {
                Console.Write(words[i].PadRight(8));
                for (int j = 0; j <= i; j++)
                {
                    Console.Write($"{attention[i][j] * 100:0}%".PadLeft(8));
                }
                Console.WriteLine();
            }
        }
    }
}