# TokenFormer - Minimal

Minimal implementation of TokenFormer for inference and learning

## Note

This code is NOT working as expected! The output s gibberish.

Feel comfortable to send a PR if you figure out what is the problem


## Running

1. Download one of the models on the [original repo](https://github.com/Haiyang-W/TokenFormer?tab=readme-ov-file#-model-zoo)

2. Run on the terminal:

```
python3 tokenformer_minimal.py --config config/150M_eval.yml --model pytorch_model.bin --prompt "Hello, how are you?"
```
