#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src/content/blog"
CACHE_DIR = ROOT / ".cache/i18n"
SOURCE_LOCALE = "zh-CN"
CONFIG_FILE = ROOT / "src/i18n/config.mjs"


def _load_supported_locales() -> tuple[str, ...]:
    """Read configured locale codes from src/i18n/config.mjs dynamically.

    Falls back to a conservative default list if parsing fails.
    """

    try:
        config_text = CONFIG_FILE.read_text(encoding="utf-8")
        locale_pattern = re.findall(r"export const locales\s*=\s*\[(.*?)\];", config_text, re.S)
        if not locale_pattern:
            raise ValueError("locales declaration not found")

        locales = re.findall(r"['\"]([^'\"]+)['\"]", locale_pattern[0])

        default_match = re.search(r"export const defaultLocale\s*=\s*['\"]([^'\"]+)['\"]", config_text)
        default_locale = default_match.group(1) if default_match else "zh-cn"

        parsed = tuple(locale for locale in locales if locale != default_locale)
        if not parsed:
            raise ValueError("no route locales found in config")
        return parsed
    except Exception:
        return ("en", "ja", "fr", "de")


SUPPORTED_LOCALES = _load_supported_locales()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_cache(locale: str) -> dict[str, str]:
    path = CACHE_DIR / f"{locale}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(locale: str, cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{locale}.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def split_frontmatter(markdown: str) -> tuple[str, str]:
    if not markdown.startswith("---\n"):
        return "", markdown
    end = markdown.find("\n---", 4)
    if end == -1:
        return "", markdown
    return markdown[4:end], markdown[end + 5 :].lstrip("\n")


def update_source_hash(frontmatter: str, source_hash: str) -> str:
    lines = frontmatter.splitlines()
    rendered = False
    output: list[str] = []

    for line in lines:
        if re.match(r"^\s*sourceHash\s*:", line):
            output.append(f"sourceHash: {json.dumps(source_hash, ensure_ascii=False)}")
            rendered = True
            continue
        output.append(line)

    if not rendered:
        output.append(f"sourceHash: {json.dumps(source_hash, ensure_ascii=False)}")

    return "\n".join(output)


def parse_yaml_string(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {"'", '"'}:
        return trimmed[1:-1]
    return trimmed


def read_source_hash(frontmatter: str) -> str | None:
    for line in frontmatter.splitlines():
        match = re.match(r"^\s*sourceHash\s*:\s*(.+)\s*$", line)
        if match:
            return parse_yaml_string(match.group(1))
    return None


def quote_yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def protect_fenced_code(text: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def replace(match: re.Match[str]) -> str:
        token = f"@@I18N_CODE_BLOCK_{len(blocks)}@@"
        blocks.append(match.group(0))
        return token

    protected = re.sub(r"(^|\n)(```|~~~)[\s\S]*?\n\2", replace, text)
    return protected, blocks


def restore_fenced_code(text: str, blocks: list[str]) -> str:
    for index, block in enumerate(blocks):
        text = text.replace(f"@@I18N_CODE_BLOCK_{index}@@", block)
    return text


def chunk_text(text: str, max_length: int = 950) -> list[str]:
    pieces = re.split(r"(\n{2,})", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_length:
            chunks.append(current)
            current = ""
        if len(piece) > max_length:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(piece), max_length):
                chunks.append(piece[i : i + max_length])
            continue
        current += piece
    if current:
        chunks.append(current)
    return chunks


def translate_chunk(text: str, locale: str, cache: dict[str, str]) -> str:
    if not text.strip():
        return text
    key = sha1(f"{locale}\n{text}")
    if key in cache:
        return cache[key]

    params = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": SOURCE_LOCALE,
            "tl": locale,
            "dt": "t",
            "q": text,
        }
    )
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 static-i18n-generator"})
    last_error: Exception | None = None

    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            translated = "".join(segment[0] for segment in data[0] if segment and segment[0])
            cache[key] = translated
            time.sleep(0.06)
            return translated
        except Exception as error:  # noqa: BLE001 - retry network translation.
            last_error = error
            time.sleep(0.25 * attempt)

    raise RuntimeError(f"failed to translate chunk for {locale}: {last_error}")


def translate_text(text: str, locale: str, cache: dict[str, str]) -> str:
    protected, blocks = protect_fenced_code(text)
    translated = "".join(translate_chunk(chunk, locale, cache) for chunk in chunk_text(protected))
    return restore_fenced_code(translated, blocks)


def translate_frontmatter(frontmatter: str, locale: str, cache: dict[str, str]) -> str:
    output: list[str] = []
    for line in frontmatter.splitlines():
        match = re.match(r"^(\s*(title|description)\s*:\s*)(.*)$", line)
        if match:
            prefix, _, raw_value = match.groups()
            value = parse_yaml_string(raw_value)
            output.append(f"{prefix}{quote_yaml_string(translate_text(value, locale, cache))}")
            continue

        tags_match = re.match(r"^(\s*tags\s*:\s*)(\[.*\])\s*$", line)
        if tags_match:
            prefix, raw_tags = tags_match.groups()
            try:
                tags = json.loads(raw_tags)
            except json.JSONDecodeError:
                output.append(line)
                continue
            translated_tags = [
                translate_text(str(tag), locale, cache) if isinstance(tag, str) else tag
                for tag in tags
            ]
            output.append(f"{prefix}{json.dumps(translated_tags, ensure_ascii=False)}")
            continue

        output.append(line)
    return "\n".join(output)


def translate_file(source_path: Path, locale: str, cache: dict[str, str], force: bool) -> str:
    output_dir = SOURCE_DIR / locale
    output_path = output_dir / source_path.name

    source_text = source_path.read_text(encoding="utf-8")
    frontmatter, body = split_frontmatter(source_text)
    source_hash = sha1(source_text)

    if output_path.exists() and not force:
        translated_frontmatter, _ = split_frontmatter(output_path.read_text(encoding="utf-8"))
        existing_hash = read_source_hash(translated_frontmatter)
        if existing_hash == source_hash:
            return "up_to_date"

    translated_frontmatter = translate_frontmatter(frontmatter, locale, cache)
    translated_frontmatter = update_source_hash(translated_frontmatter, source_hash)
    translated_body = translate_text(body, locale, cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"---\n{translated_frontmatter}\n---\n\n{translated_body}", encoding="utf-8")
    return "updated"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local SSG Markdown translations for blog posts.")
    parser.add_argument("--locales", default=",".join(SUPPORTED_LOCALES), help="Comma-separated target locales.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated translations.")
    args = parser.parse_args()

    locales = [locale.strip() for locale in args.locales.split(",") if locale.strip()]
    for locale in locales:
        if locale not in SUPPORTED_LOCALES:
            raise SystemExit(f"Unsupported locale: {locale}")

    source_paths = sorted(path for path in SOURCE_DIR.glob("*.md") if path.is_file())
    for locale in locales:
        cache = load_cache(locale)
        counts = {"updated": 0, "up_to_date": 0}
        for source_path in source_paths:
            status = translate_file(source_path, locale, cache, args.force)
            counts[status] += 1
            print(f"[{locale}] {status}: {source_path.name}", flush=True)
        save_cache(locale, cache)
        print(
            f"[{locale}] done: {counts['updated']} updated, {counts['up_to_date']} unchanged",
            flush=True,
        )


if __name__ == "__main__":
    main()
