# Divergent Inference Architecture (DIA)

A work-in-progress tool for exploring the semantic branching behavior of language models, particularly when they encounter prompts that might lead to conflicting lines of reasoning.

## What This Does

Most language model interfaces show you the single "best" continuation from any given prompt. DIA takes a different approach: it generates many possible continuations, clusters them semantically, and builds a tree structure that reveals where the model's reasoning genuinely diverges.

This is based on the hypothesis that consumer-tuned models may still contain latent capabilities for multiple, semantically conflicting inference chains - even if those chains are buried under safety training or preference optimization. By sampling broadly and clustering semantically similar continuations, we can potentially surface these alternative reasoning paths.

## Current State

This is very much a work in progress. The basic pipeline functions and can:

- Generate multiple continuation "stems" from branching points
- Use sentence embeddings to cluster semantically similar continuations
- Build an interactive tree visualization showing where semantic divergence is detected

## Limitations and TODOs

- Only tested on GPT-2 models, which aren't really intelligent enough to function as an actual proof of concept. (larger models will need better memory management)
- Clustering parameters need tuning for different domains. Dynamic adjustments to sample size based on mean embedding vector distance would probably be a good idea
- Currently no quantitative metrics for "semantic divergence"
- Visualization is bare-bones and could use more features/polish
- Data output is sparse at the moment. Could use JSON reports with more comprehensive stats.
- Limited testing on different model types and prompt categories

This is experimental software. If you find it useful or have ideas for improvement, contributions are welcome.

## Possible Use Cases

- **Security research**: Identifying potential jailbreak vectors or unintended model behaviors
- **Model evaluation**: Understanding how alignment training affects latent model capabilities
- **Curiosity**: Exploring the "what if" space around contentious or ambiguous prompts

## Quick Start

TODO - add requirements.txt and quick start instructions

## Configuration

Edit `config.ini` to adjust:

- Model settings (currently supports GPT-2 variants)
- Generation parameters (temperature, top-k, top-p)
- Clustering sensitivity (eps, minimum cluster size)
- Visualization options
TODO - more detail about config settings

## Architecture

- `model_interface.py`: Abstraction layer for different language models
- `embedding_analyzer.py`: Semantic clustering using sentence transformers
- `tree_utils.py`: Core tree data structures and operations
- `visualization.py`: Interactive HTML tree visualization
- `divergent.py`: Main generation pipeline

## Why This Matters

Language models are increasingly deployed as single-response systems, but their training creates rich internal landscapes of possible continuations. Understanding these landscapes - especially where they contain conflicting reasoning patterns - seems important for both safety and capability assessment.

Whether this particular approach actually helps with that remains to be seen. Further testing and development is required.
