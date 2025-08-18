import torch
from divergent import DivergentGenerator


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    generator = DivergentGenerator()
    test_prompts = ["In the question of individual autonomy versus collective welfare, I'd argue the more ethical position is"]

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*70}")

        # Generate tree and get complete analysis
        analysis = generator.full_analysis(prompt, print_stems=False)

        # Export visualization
        html_path = generator.visualizer.quick_export(analysis['root'], prompt)
        print(f"Visualization saved to: {html_path}")

        # Print analysis results
        print(f"\nBranching ratio: {analysis['branching_ratio']:.2f}")
        print(f"Average path length: {analysis['average_path_length']:.1f}")

        generator.printer.print_sample_paths(
            analysis['root'],
            num_paths=3,
            prompt=prompt
        )


if __name__ == "__main__":
    main()
