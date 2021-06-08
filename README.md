# Variational Auto Encoder (VAE)

Non-official implementation of the paper [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114?fileGuid=WyYwxqq8kWjKdWgd).

## Installation

### Manual

```bash
python -m venv venv

# Linux/Mac way of activating python virtual environnements
source venv/bin/activate
# Windows way of activating python virtual environnements
venv/Scripts/activate.bat

pip install -r requirements.txt
jupyter lab
```

### Portable

Add the flag -W for each if you are on Windows.

```bash
./helper.sh install
./helper.sh launch
```

### Docker

```bash
docker build -t saundersp/vae .
docker run -p 8888:8888 saundersp/vae
```

_2021 Pierre Saunders @saundersp_
