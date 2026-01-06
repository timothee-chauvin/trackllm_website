"""Download tokenizers from HuggingFace and save token ID to UTF-8 string mappings."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

from huggingface_hub import login
from transformers import AutoTokenizer

from trackllm_website.config import config, logger

tokenizers_dir = config.bi.tokenizers_dir
index_path = tokenizers_dir / "index.csv"


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


def get_tokenizer_hash(vocab: dict[int, str]) -> str:
    """Get a hash of the vocabulary to identify unique tokenizers."""
    items = sorted(vocab.items())
    content = "\n".join(f"{k}\t{v}" for k, v in items)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def download_tokenizer(repo: str) -> dict[int, str] | None:
    """Download tokenizer and extract token ID to string mapping."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {repo}: {e}")
        return None

    vocab: dict[int, str] = {}

    try:
        token_to_id = tokenizer.get_vocab()
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

    for token_str, token_id in token_to_id.items():
        if token_str in special_tokens or token_id in special_ids:
            continue

        # Skip tokens that look like special tokens (e.g., <|endoftext|>, <pad>)
        if token_str.startswith("<") and token_str.endswith(">"):
            continue

        try:
            decoded = tokenizer.convert_tokens_to_string([token_str])
            vocab[token_id] = decoded
        except Exception:
            vocab[token_id] = token_str

    return vocab


def save_tokenizer_csv(vocab: dict[int, str], path: Path) -> None:
    """Save vocabulary as CSV with token ID and UTF-8 string columns."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "token_string"])
        for token_id in sorted(vocab.keys()):
            writer.writerow([token_id, vocab[token_id]])


def load_existing_index() -> dict[str, str]:
    """Load existing index.csv if it exists."""
    if not index_path.exists():
        return {}
    existing: dict[str, str] = {}
    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing[row["model"]] = row["tokenizer_hash"]
    return existing


def save_index(model_to_hash: dict[str, str]) -> None:
    """Save index.csv with current model to hash mappings."""
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "tokenizer_hash"])
        for model in sorted(model_to_hash.keys()):
            writer.writerow([model, model_to_hash[model]])


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
            tokenizer_path = tokenizers_dir / f"{existing_index[model]}.csv"
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
            existing_path = tokenizers_dir / f"{vocab_hash}.csv"
            if existing_path.exists():
                logger.info(f"  -> Tokenizer already on disk (hash: {vocab_hash})")
            else:
                logger.info(
                    f"  -> New tokenizer with {len(vocab)} tokens (hash: {vocab_hash})"
                )
                save_tokenizer_csv(vocab, existing_path)
                logger.info(f"  Saved {existing_path.name}")
            saved_hashes.add(vocab_hash)

        model_to_hash[model] = vocab_hash
        save_index(model_to_hash)

    logger.info(f"Saved index to {tokenizers_dir / 'index.csv'}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"SUMMARY: {len(model_to_hash)} succeeded, {len(failed_models)} failed")
    logger.info(f"{'=' * 60}")
    if failed_models:
        logger.info("Models without tokenizers found:")
        for m in failed_models:
            logger.info(f"  - {m}")


def load_tokenizer_vocab(tokenizer_hash: str) -> list[str]:
    """Load a tokenizer's vocabulary in order."""
    path = tokenizers_dir / f"{tokenizer_hash}.csv"
    strings: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strings.append(row["token_string"])
    return strings


def get_best_single_token_strings() -> list[str]:
    """Return strings sorted by frequency across known tokenizers (descending).

    Strings that appear as single tokens in more tokenizers are ranked higher,
    as they're more likely to be single tokens in unknown tokenizers.
    """
    index = load_existing_index()
    unique_hashes = set(index.values())

    string_counts: Counter[str] = Counter()
    for tok_hash in unique_hashes:
        vocab = set(load_tokenizer_vocab(tok_hash))
        string_counts.update(vocab)

    # Sort by count descending, then by byte length ascending, then alphabetically
    # The alphabetical sort is to make this function deterministic.
    items = string_counts.most_common()
    items.sort(key=lambda x: (-x[1], len(x[0].encode("utf-8")), x[0]))
    return [item[0] for item in items]


if __name__ == "__main__":
    main()
