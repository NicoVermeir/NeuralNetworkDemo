// The four nights from the slides, and what a brain decides on each one
static class Party
{
    static readonly (string Night, double Alice, double Bob)[] Nights =
        [
            ("nobody", 0, 0), 
            ("Alice", 1, 0), 
            ("Bob", 0, 1), 
            ("both", 1, 1)
        ];

    public static void Show(Func<double, double, double> brain)
    {
        foreach (var (night, alice, bob) in Nights)
        {
            double answer = brain(alice, bob);
            string decision = answer >= 0.5 ? "go" : "home";
            Console.WriteLine($"{night,-7} {decision,-5} {answer:0.00}");
        }

        Console.WriteLine();
    }
}