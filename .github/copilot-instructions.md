# AI Development Project Guidelines

This repository contains multiple AI/ML projects organized in phases, focusing on different aspects of artificial intelligence and machine learning. Here's what you need to know to work effectively with this codebase.

## Project Structure

The repository is organized into phases:

```
Fase_1/ - Fundamentals & Computer Vision
Fase_2/ - Natural Language Processing & Genetic Algorithms
Fase_3/ - Fine-tuning & LangChain
Fase_4/ - Advanced Computer Vision
```

## Key Technologies & Dependencies

### Common Python Dependencies
- transformers
- torch
- unsloth (for model optimization)
- langchain
- datasets
- deepeval (for model evaluation)

### Environment Setup
Most projects require Python 3.7+ and specific package versions. Check individual project READMEs for specific requirements.

## Major Components

### Fine-tuning Workflow (`Fase_3/`)
- Uses Unsloth for optimized model training
- Implements LoRA adapters for efficient fine-tuning
- Key files: `finetunning.py`, `FineTunning - Rodando Maquina Mais Basica.ipynb`

Example pattern for fine-tuning:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b",
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = True
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj","v_proj"],
    lora_alpha = 32
)
```

### Computer Vision Projects (`Fase_1/Visao_Computacional/`, `Fase_4/`)
- Implements facial recognition, object detection, and pose estimation
- Uses various pre-trained models and custom implementations
- Focus on practical applications and real-time processing

### LangChain Integration (`Fase_3/4 - LangChain/`)
Example agent structure:
```python
# Common pattern for agent creation
def criar_chain(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # Add configuration here
    )
```

## Development Workflows

### Model Training
1. Data Preparation:
   - Clean and structure input data
   - Format prompts using templates (e.g., Alpaca-style)
2. Training Configuration:
   - Set model parameters and LoRA adaptors
   - Configure training arguments
3. Evaluation:
   - Use deepeval for model assessment
   - Test with domain-specific prompts

### Environment Management
- Each phase/project typically has its own requirements.txt
- Use virtual environments for isolation
- GPU support recommended for training

## Project-Specific Conventions

### Fine-tuning Projects
- Models are saved with LoRA adapters separately
- Use Google Drive for model storage in Colab environments
- Follow the Alpaca prompt template for consistency

### Computer Vision Projects
- Organize processing pipelines into distinct steps
- Use common utilities for video/image processing
- Cache processed results when appropriate

## Integration Points

### Model Loading
```python
base_model_name = "unsloth/Phi-4-mini-instruct-bnb-4bit"  # Common choice for smaller models
model, tokenizer = FastLanguageModel.from_pretrained(
    base_model_name,
    # Configuration follows project standards
)
```

### Data Flow
1. Raw Data → Preprocessing → Training Format
2. Training → Model Adaptation → Evaluation
3. Deployment → Integration → Monitoring

## Debugging Tips
- Check CUDA memory usage for GPU-based training
- Monitor training metrics with tensorboard
- Use deepeval for systematic testing
- Review model outputs against expected patterns