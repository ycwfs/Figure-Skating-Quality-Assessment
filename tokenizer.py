from iopath.common.file_io import g_pathmgr
import io
import gzip
import html
import math
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import ftfy
import numpy as np
import regex as re
import torch
import torch.nn as nn

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text    

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str, context_length=100):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with g_pathmgr.open(bpe_path, "rb") as fh:
            bpe_bytes = io.BytesIO(fh.read())
            merges: List[str] = gzip.open(bpe_bytes).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges: List[Tuple[str, ...]] = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>", "<video>", "<audio>", "<tes>", "<pcs>","<pad>"])
        self.vacab_size = len(vocab)
        # 0 - 49412
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
            "<video>": "<video>",
            "<audio>": "<audio>",
            "<tes>": "<tes>",
            "<pcs>": "<pcs>",
            "<pad>": "<pad>"
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|<video>|<audio>|<tes>|<pcs>|<pad>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.context_length = context_length

        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]
        self.video_token = self.encoder["<video>"]
        self.audio_token = self.encoder["<audio>"]
        self.tes_token = self.encoder["<tes>"]
        self.pcs_token = self.encoder["<pcs>"]
        self.pad_token = self.encoder["<pad>"]

    def get_vocab_size(self):
        return self.vacab_size

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def prepare_input(self, texts, vf, af, context_length=None):
        #assert vf.shape == af.shape

        if not context_length:
            context_length = self.context_length

        clips = vf.shape[0]
        result = torch.zeros(context_length, dtype=torch.long)
        res =[]
        res.extend(clips * [self.video_token,self.audio_token] + [self.tes_token, self.pcs_token])
        res.extend([self.sot_token] + self.encode(texts) + [self.eot_token])

        if len(res) < context_length:
            result[:len(res)] = torch.tensor(res)
            result[len(res):] = self.pad_token
        else:
            result = torch.tensor(res[:context_length])

        return result

if __name__ == '__main__':
    import numpy as np
    t = SimpleTokenizer("/data1/1/code/helping/ImageBind/imagebind/bpe/bpe_simple_vocab_16e6.txt.gz")
    print(t.prepare_input('How this figure skating competition perform? In this women short program competition, it can get a total element score of 38.67 and a total program component score of 33.88.',np.array([1,2,3]),np.array([1,2,3])))