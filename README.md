# PicoTuner

PicoTuner offers a versatile solution for fine-tuning Mistral, Llama2, CodeLLama models, including their hefty 70B/35B counterparts, on Apple's M1/M2 devices (e.g., Macbook Air, Mac Mini) and consumer Nvidia GPUs.

## Overview

PicoTuner is built for developers looking to fine-tune large language models on hardware that's more readily available to consumers. Unlike typical approaches that may employ quantization, PicoTuner uses a novel method that leverages SSD or main memory during both the forward and backward passes. This allows for efficient model fine-tuning without the need for high-end, enterprise-level hardware.

### Key Features

- **No Quantization**: Utilizes [_slowllama_](https://github.com/okuvshynov/slowllama) technique for offloading parts of the model to persistent storage or main memory, optimizing for fine-tuning performance over interactivity.
- **LoRA Implementation**: Employs Learning without Forgetting (LoRA) to constrain updates to a smaller subset of model parameters, enhancing the fine-tuning process.
- **Apple Silicon & CUDA Support**: Compatible with both Apple M1/M2 devices and Nvidia GPUs, with specialized experimental support for CUDA environments.
- **Dedicated to Fine-Tuning**: The project is focused exclusively on model fine-tuning, without any specific optimizations for inference. For inference needs, refer to the [_llama.cpp_](https://github.com/ggerganov/llama.cpp) implementation.

## Motivation

While training large models from scratch is often beyond reach due to resource constraints, fine-tuning pre-trained models remains a practical approach. PicoTuner is designed to make fine-tuning accessible for developers with limited hardware capabilities, letting you achieve meaningful model improvements over time.

## Experimental Status

PicoTuner is experimental, even more so for CUDA. Extensive experimentation, including a dedicated report on the A10 GPU, has laid the groundwork for ongoing improvements and optimizations.

## Mistral Models Integration

In addition to Llama2 and CodeLLama models, PicoTuner is also integrated with Mistral models. This integration enhances the fine-tuning capabilities of PicoTuner, providing users with more options to tailor their models to specific tasks and datasets.

## Getting Started

To start fine-tuning your models with PicoTuner, please refer to the documentation for detailed instructions on setup, configuration, and usage.

### Setup

(1) Make sure you have xcode installed... at least the command line parts

```bash
# check the path of your xcode install
xcode-select -p

# xcode installed returns
# /Applications/Xcode-beta.app/Contents/Developer

# if xcode is missing then install it... it takes ages;
xcode-select --install
```

(2) Install the conda version for MacOS that supports Metal GPU

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```

(3) Make a conda environment

```bash
conda create -n pico python=3.11.6
conda activate pico
```

(4) Clone git repo

```bash
git clone https://github.com/Gincioks/PicoTuner
cd PicoTuner
```

(5) Install the LATEST torch version and pip packages

```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt
```

(6) Run `./pico.py init` and select model. Mistral models should be downloaded automatic.

Models folder stucture:
models/{model_collection}/{model}/{base;prepared;finetuned}

Model weigths put in base folder. Folder should have atleast 3 file: params.json, tokenizer.model, consolidated.\*\*.pth

## Contributions

Contributions are welcome! If you have ideas for improvements or have found a bug, please open an issue or submit a pull request.

## License

PicoTuner is open-source software licensed under the MIT license.

TODO:

Model args: Needs change

repeat_kv: Needs change

Attention: Needs change
