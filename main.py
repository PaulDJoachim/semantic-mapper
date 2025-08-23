from models.model_interface import create_generator


def main():
    """Production execution with real models."""
    generator = create_generator(model_type="gpt2")
    
    test_prompts = [
        "In the question of individual autonomy versus collective welfare, I'd argue the more ethical position is"
    ]

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*70}")

        analysis = generator.full_analysis(prompt, print_stems=False)
        html_path = generator.visualizer.quick_export(analysis['root'], prompt)
        
        print(f"Visualization: {html_path}")
        print(f"Branching ratio: {analysis['branching_ratio']:.2f}")
        print(f"Average path length: {analysis['average_path_length']:.1f}")

        generator.printer.print_sample_paths(
            analysis['root'],
            num_paths=3,
            prompt=prompt
        )


if __name__ == "__main__":
    main()