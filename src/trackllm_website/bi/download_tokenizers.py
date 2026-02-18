"""Download tokenizers from HuggingFace and save unique token strings."""

import base64
import hashlib
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import orjson
from huggingface_hub import login
from transformers import AutoTokenizer

from trackllm_website.config import config, logger

tokenizers_dir = config.bi.tokenizers_dir
index_path = tokenizers_dir / "index.json"


def openrouter_to_hf(model: str) -> str | None:
    """Convert OpenRouter model ID to HuggingFace repo path.

    Returns None if no mapping is known.
    """
    parts = model.split("/", 1)
    if len(parts) != 2:
        return None

    provider, model_name = parts

    if provider == "deepseek":
        # deepseek/deepseek-v3 -> deepseek-ai/deepseek-v3
        return f"deepseek-ai/{model_name}"

    if provider == "meta-llama":
        # meta-llama/llama-3-70b-instruct -> meta-llama/meta-llama-3-70b-instruct
        if model_name.startswith("llama-3-"):
            return f"meta-llama/meta-{model_name}"
        return model  # already correct format

    if provider == "qwen":
        # qwen/qwen-2.5-72b-instruct -> qwen/qwen2.5-72b-instruct
        # Remove dash between qwen and version number
        if model_name.startswith("qwen-"):
            return f"qwen/{model_name.replace('qwen-', 'qwen', 1)}"
        return model

    return None


def setup_hf_auth() -> None:
    """Login to HuggingFace using token from environment."""
    token = config.hf_token
    if not token:
        raise ValueError("HF_TOKEN is not set in the configuration")
    login(token=token)
    logger.info("Logged in to HuggingFace")


def get_tokenizer_hash(vocab: list[str]) -> str:
    """Get a hash of the vocabulary list to identify unique tokenizers."""
    # Vocab is expected to be a sorted list of strings here
    content = "\n".join(vocab)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def download_tokenizer(repo: str) -> list[str] | None:
    """Download tokenizer and extract unique token strings.

    Returns a sorted list of unique strings found in the vocabulary.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {repo}: {e}")
        return None

    try:
        token_map = tokenizer.get_vocab()
    except Exception as e:
        logger.warning(f"Failed to get vocab from {repo}: {e}")
        return None

    # Get special tokens to exclude
    special_tokens: set[str] = set()
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens.update(tokenizer.all_special_tokens)
    if hasattr(tokenizer, "additional_special_tokens"):
        special_tokens.update(tokenizer.additional_special_tokens or [])

    special_ids: set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    vocab_set: set[str] = set()
    for token_str, token_id in token_map.items():
        if token_str in special_tokens or token_id in special_ids:
            continue
        # Skip tokens that look like special tokens (e.g., <|endoftext|>, <pad>)
        if token_str.startswith("<") and token_str.endswith(">"):
            continue

        # make sure we don't have too much weird stuff in there
        assert token_str.encode("utf-8").decode("utf-8") == token_str

        if token_str.startswith("Ġ") or token_str.startswith("▁"):
            token_str = " " + token_str[1:]

        # make sure the strings get encoded as a single token
        tokenized = tokenizer.encode(token_str)
        if (
            len(tokenized) == 0
            or len(tokenized) > 2
            or (len(tokenized) == 2 and tokenized[0] not in special_ids)
        ):
            continue

        vocab_set.add(token_str)

    # Return sorted list for deterministic hashing
    return sorted(list(vocab_set))


def save_tokenizer_json(vocab: list[str], path: Path) -> None:
    """Save vocabulary list as JSON with base64 encoded strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab_b64 = [base64.b64encode(token.encode()).decode() for token in vocab]
    with open(path, "wb") as f:
        f.write(orjson.dumps(vocab_b64))


def load_existing_index() -> dict[str, str]:
    """Load existing index.json if it exists."""
    if not index_path.exists():
        return {}
    with open(index_path, "rb") as f:
        return orjson.loads(f.read())


def save_index(model_to_hash: dict[str, str]) -> None:
    """Save index.json with current model to hash mappings."""
    with open(index_path, "wb") as f:
        f.write(orjson.dumps(model_to_hash, option=orjson.OPT_SORT_KEYS))


def main() -> None:
    tokenizers_dir.mkdir(parents=True, exist_ok=True)

    setup_hf_auth()

    existing_index = load_existing_index()

    # Collect unique models from endpoints
    models = sorted({ep.model for ep in config.endpoints_bi})
    logger.info(f"Found {len(models)} unique models")

    failed_models: list[str] = []
    model_to_hash: dict[str, str] = dict(existing_index)
    saved_hashes: set[str] = set()

    for model in models:
        if model in existing_index:
            tokenizer_path = tokenizers_dir / f"{existing_index[model]}.json"
            if tokenizer_path.exists():
                logger.info(f"Skipping {model} (already downloaded)")
                continue

        logger.info(f"Trying tokenizer: {model}")

        # Try HuggingFace-mapped name first, then fall back to original
        hf_repo = openrouter_to_hf(model)
        vocab = None
        if hf_repo:
            logger.info(f"  -> Mapped to HF repo: {hf_repo}")
            vocab = download_tokenizer(hf_repo)
        if vocab is None:
            vocab = download_tokenizer(model)
        if vocab is None:
            failed_models.append(model)
            continue

        vocab_hash = get_tokenizer_hash(vocab)

        if vocab_hash in saved_hashes:
            logger.info(f"  -> Same as existing tokenizer (hash: {vocab_hash})")
        else:
            existing_path = tokenizers_dir / f"{vocab_hash}.json"
            if existing_path.exists():
                logger.info(f"  -> Tokenizer already on disk (hash: {vocab_hash})")
            else:
                logger.info(
                    f"  -> New tokenizer with {len(vocab)} unique strings (hash: {vocab_hash})"
                )
                save_tokenizer_json(vocab, existing_path)
                assert load_tokenizer_vocab(vocab_hash) == vocab
                logger.info(f"  Saved {existing_path.name}")
            saved_hashes.add(vocab_hash)

        model_to_hash[model] = vocab_hash
        save_index(model_to_hash)

    logger.info(f"Saved index to {tokenizers_dir / 'index.json'}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY: {len(model_to_hash)} succeeded, {len(failed_models)} failed")
    logger.info(f"{'=' * 60}")
    if failed_models:
        logger.info("Models without tokenizers found:")
        for m in failed_models:
            logger.info(f"  - {m}")


@lru_cache(maxsize=None)
def load_tokenizer_vocab(tokenizer_hash: str, shuffle: bool = True) -> list[str]:
    """Load a tokenizer's vocabulary list.

    If shuffle is True, the strings are shuffled. The output of this function
    is deterministic even with shuffle=True.
    """
    path = tokenizers_dir / f"{tokenizer_hash}.json"
    with open(path, "rb") as f:
        vocab_b64 = orjson.loads(f.read())
        vocab = [base64.b64decode(token).decode() for token in vocab_b64]
        if shuffle:
            random.Random(0).shuffle(vocab)
        return vocab


def get_best_single_token_strings(shuffle: bool = True) -> list[str]:
    """Return strings sorted by frequency across known tokenizers (descending).

    Strings that appear as single tokens in more tokenizers are ranked higher,
    as they're more likely to be single tokens in unknown tokenizers.

    If shuffle is True, for a given frequency, the strings are shuffled. The output of this function
    is deterministic given the available tokenizers, even with shuffle=True.
    """
    index = load_existing_index()
    unique_hashes = set(index.values())

    string_counts: Counter[str] = Counter()
    for tok_hash in unique_hashes:
        vocab = load_tokenizer_vocab(tok_hash, shuffle=False)
        string_counts.update(vocab)

    # Group by count, maybe shuffle within each group, then sort by count descending
    count_to_strings: dict[int, list[str]] = {}
    for string, count in string_counts.items():
        count_to_strings.setdefault(count, []).append(string)

    result: list[str] = []
    for count in sorted(count_to_strings.keys(), reverse=True):
        strings = count_to_strings[count]
        if shuffle:
            random.Random(0).shuffle(strings)
        result.extend(strings)
    assert len(set(result)) == len(result)
    return result


if __name__ == "__main__":
    main()
