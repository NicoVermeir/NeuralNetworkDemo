using NeuralNetworkDemo.Model;

namespace NeuralNetworkDemo.Chapters;

public static class Chapter4
{
    internal static void Run()
    {
        // Every name becomes lessons: three letters in, the next letter out
        var lessons = new List<(string Context, char Next)>();
        foreach (string name in File.ReadAllLines("names.txt"))
        {
            string context = "...";
            foreach (char next in name + ".")
            {
                lessons.Add((context, next));
                context = context[1..] + next;  // slide the window one letter
            }
        }
        Console.WriteLine($"{lessons.Count} lessons");

        // 81 switches -> 100 Sigmoid neurons -> 27 letter scores
        var random = new Random(42);
        var model = new Network(
          Layer.CreateRandom(81, 100, random, Activation.Sigmoid),  // 81 in, 100 out
          Layer.CreateRandom(100, 27, random, Activation.Identity)); // 100 in, 27 out

        // The dice that pick the letters
        var dice = new Random(7);

        string Names(int count, double temperature = 1) => string.Join(", ", Enumerable.Range(0, count).Select(_ => Generate(model, dice, temperature)));

        Console.WriteLine($"Untrained: {Names(5)}");

        //Train(model, lessons, random); 
        model.Load("names-model.txt");

        Console.WriteLine($"Trained:  {Names(10)}");
        Console.WriteLine($"T = 0.5:  {Names(6, 0.5)}");
        Console.WriteLine($"T = 2:   {Names(6, 2)}");

        model.Save("names-model.txt");
    }

    // One letter at a time, until the model picks the dot
    public static string Generate(Network model, Random dice, double temperature = 1)
    {
        string name = "", context = "...";
        while (name.Length < 20)  // a hot model rambles, so cap it
        {
            double[] scores = model.Forward(Letters.OneHot(context));
            double[] chances = Neuron.Softmax(scores, temperature);
            char next = Letters.Alphabet[Pick(chances, dice)];

            if (next == '.')
            {
                break;  // the model says the name is done
            }

            name += next;
            context = context[1..] + next;
        }
        return name;
    }

    // Roll a loaded die: letter i comes up with chance chances[i]
    static int Pick(double[] chances, Random dice)
    {
        double roll = dice.NextDouble(), total = 0;
        for (int i = 0; i < chances.Length; i++)
        {
            total += chances[i];
            if (roll < total)
            {
                return i;
            }
        }

        return chances.Length - 1;
    }

    static void Train(Network model, List<(string Context, char Next)> lessons, Random random)
    {
        double learningRate = 0.1, lossSum = 0;
        int count = 0;

        for (int step = 1; step <= 200_000; step++)
        {
            // One random lesson: three letters in, the right next letter
            var (context, next) = lessons[random.Next(lessons.Count)];
            int answer = Letters.IndexOf(next);

            // Forward: decide. Loss: how surprised by the right letter?
            double[] scores = model.Forward(Letters.OneHot(context));
            double[] chances = Neuron.Softmax(scores);
            lossSum += -Math.Log(chances[answer]);
            count++;

            var blame = new double[27];
            for (int i = 0; i < 27; i++)
            {
                double should = i == answer ? 1 : 0;
                blame[i] = chances[i] - should;
            }
            model.Backward(blame, learningRate);

            if (step == 1_000 || step % 20_000 == 0)
            {
                double surprise = lossSum / count;
                Console.WriteLine($"step {step,7}  surprise {surprise:0.00}");
                lossSum = 0;
                count = 0;
            }
        }
    }
}
