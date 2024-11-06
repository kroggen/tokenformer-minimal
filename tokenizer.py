from abc import ABC, abstractmethod
from tokenizers import Tokenizer
from typing import List, Union


def build_tokenizer(tokenizer_type: str, vocab_file: str = None, padding_multiple: int = 128):
    """Initialize tokenizer.
    
    Args:
        tokenizer_type: Type of tokenizer to build
        vocab_file: Path to vocabulary file (required for HFTokenizer)
    """
    print(f"Building tokenizer...", flush=True)

    # Select and instantiate the tokenizer.
    if tokenizer_type.lower() == "HFTokenizer".lower():
        assert vocab_file is not None, "vocab_file is required for HFTokenizer"
        tokenizer = HFTokenizer(vocab_file)
    else:
        raise NotImplementedError(
            f"{tokenizer_type} tokenizer is not implemented."
        )

    # Add padded vocab size
    tokenizer.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, padding_multiple)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, padding_multiple):
    """Pad vocab size so it is divisible by padding_multiple."""
    after = orig_vocab_size
    while (after % padding_multiple) != 0:
        after += 1
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} " "tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )

class HFTokenizer(AbstractTokenizer):
    """Designed to Integrate HF's Tokenizer library."""

    def __init__(self, vocab_file):
        name = "HFTokenizer"
        super().__init__(name)
        self.tokenizer = Tokenizer.from_file(vocab_file)
        self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = self.tokenizer.token_to_id("<|padding|>")

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text: str):
        return self.tokenizer.encode(text).ids

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
