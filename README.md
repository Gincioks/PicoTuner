## Setup

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
git clone
cd pico_tuner
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

TODO:
Model args | Needs change
repeat_kv | Needs change
Attention | Needs change
