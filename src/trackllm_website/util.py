import hashlib


def slugify(s: str, max_length: int = 1000, hash_length: int = 0) -> str:
    """
    Convert a string to a slugified version suitable for Linux and MacOS filenames.

    Special characters are hex-encoded to preserve information while keeping
    the filename safe. For example, "|" becomes "7c".

    Args:
        s: The input string to slugify
        max_length: Maximum length of the output without the hash
        hash_length: Length of the hash to append to the output

    Returns:
        A slugified string safe for use as a Linux or MacOS filename
    """
    slug = ""

    for char in s:
        if char.isalnum() or char in "._-+=@~,":
            slug += char
        elif char == " ":
            slug += "-"
        else:
            slug += f"{ord(char):02x}"

    slug = slug[:max_length]

    if hash_length > 0:
        string_hash = hashlib.md5(s.encode("utf-8")).hexdigest()[:hash_length]
        slug += "_" + string_hash

    return slug
