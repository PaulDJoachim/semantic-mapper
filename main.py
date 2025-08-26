from models.model_interface import create_generator
from config.config import get_config
from visualization.visualization import TreeVisualizer, TreePrinter


def main():
    config = get_config()

    model_name = config.get('model', 'model_name', 'mock')
    embedding_model = config.get('embeddings', 'model', 'mock')
    cluster_type = config.get('clustering', 'cluster_type', 'mock')

    generator = create_generator(model_name=model_name,
                                 embedding_model=embedding_model,
                                 cluster_type=cluster_type)

    analysis_output_dir = config.get('analysis', 'output_dir')

    template_path = config.get('visualization', 'template_path')
    render_output_dir = config.get('visualization', 'output_dir')
    visualizer = TreeVisualizer(template_path, render_output_dir)
    printer = TreePrinter()


    test_prompts = [
        "In the question of individual autonomy versus collective welfare, I'd argue the more ethical position is"]

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*70}")

        # Generate analysis report
        report = generator.full_analysis(prompt, print_stems=False)

        # Save and visualize
        json_path = report.save_json(analysis_output_dir)
        html_path = visualizer.quick_export(report)

        print(f"Analysis saved to: {json_path}")
        print(f"Visualization saved to: {html_path}")

        # Print summary
        printer.print_statistics(report)
        printer.print_sample_paths(report)


if __name__ == "__main__":
    main()