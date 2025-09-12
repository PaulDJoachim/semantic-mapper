from models.model_interface import create_generator
from config.config import get_config
from visualization.visualization import TreeVisualizer, TreePrinter


def main():
    config = get_config()

    inference_model = config.get('model', 'model_name')
    embedding_model = config.get('embeddings', 'model_name')
    cluster_type = config.get('clustering', 'cluster_type')

    generator = create_generator(inference_model=inference_model,
                                 embedding_model=embedding_model,
                                 cluster_type=cluster_type)

    analysis_output_dir = config.get('analysis', 'output_dir')
    render_output_dir = config.get('visualization', 'output_dir')

    tree_visualizer = TreeVisualizer()
    printer = TreePrinter()

    test_prompts = [
        # "The primary purpose of education is to",
        # "Blue whales are the largest animals on Earth, but did you know",
        # "If all A are B, and all B are C,",
        # "Remember that guy we met yesterday? I just found out",
        # "The frequency of a word is inversely proportional to",
        # "Janet my dear, I have a terrible secret I must confess - something so profound and horrific, that I'm afraid it may shatter our love like those cool pumpkin-slinging trebuchets do to the various gourds they launch. There's no easy way to say this... Janet, the truth is",
        # "Peanut butter cookies are a delicious autumn treat! Here's a recipe that you can try at home. Start by",
        # "In the question of individual autonomy versus collective welfare, I'd argue the more ethical position is",
        "Out of all the animals that have ever existed, my favorite one is undoubtedly the",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*70}")

        # Generate analysis report
        report = generator.full_analysis(prompt)

        # Save and visualize
        json_path = report.save_json(analysis_output_dir)
        html_path = tree_visualizer.export(report, render_output_dir)

        print(f"Analysis saved to: {json_path}")
        print(f"visualization saved to: {html_path}")

        # Print summary
        printer.print_statistics(report)
        printer.print_sample_paths(report)


if __name__ == "__main__":
    main()