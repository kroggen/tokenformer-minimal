# TokenFormer - Minimal

Minimal implementation of TokenFormer for inference and learning

The `pytorch` version is on this branch. Check the `tinygrad` implementation [here](https://github.com/kroggen/tokenformer-minimal/tree/tinygrad)


# Install

```
pip install -r requirements.txt
```

## Running

1. Download one of the models on the [original repo](https://github.com/Haiyang-W/TokenFormer?tab=readme-ov-file#-model-zoo)

2. Run on the terminal:

```
python3 tokenformer_minimal.py --config config/150M_eval.yml --model pytorch_model.bin --prompt "Hello, how are you?"
```
