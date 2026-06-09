#!/usr/bin/env python3
"""
Run a chat template against a local OpenAI-compatible endpoint with an image or PDF.

Examples:
  python run_template.py messages/beleg_template.json --file Belege/1.pdf
  python run_template.py -t messages/beleg_von_barbara.json --dir Belege
  python run_template.py messages/beleg_template.json --dir Belege
  python run_template.py --file Belege/1.pdf
    (default template: messages/beleg_template.json)
    (default API key: unsloth-local-token.tk)
    (default output folder: output/)

Environment variables (optional):
  LOCAL_AI_URL       API URL (default: http://localhost:8888/v1/chat/completions)
  LOCAL_AI_API_KEY   Bearer token (overrides unsloth-local-token.tk)
  LOCAL_AI_MODEL     Model name
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import requests
except ImportError:
    requests = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_URL = "http://localhost:8888/v1/chat/completions"
DEFAULT_TEMPLATE = PROJECT_ROOT / "messages" / "beleg_template.json"
DEFAULT_TOKEN_FILE = PROJECT_ROOT / "unsloth-local-token.tk"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"
DEFAULT_CATEGORIES = (
    "electronics, appliances, clothing, footwear, furniture, home, kitchen, "
    "groceries, food, health, beauty, sports, tools, automotive, office, books, "
    "toys, jewelry, other"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}


def load_api_key(cli_value: str | None) -> str | None:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    env = os.environ.get("LOCAL_AI_API_KEY", "").strip()
    if env:
        return env
    if DEFAULT_TOKEN_FILE.is_file():
        token = DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def strip_json_comments(text: str) -> str:
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    while i < len(text):
        c = text[i]
        n = text[i + 1] if i + 1 < len(text) else ""
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and n == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
            out.append("\n")
            continue
        if c == "/" and n == "*":
            i += 2
            while i < len(text) and not (text[i] == "*" and i + 1 < len(text) and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_template(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(strip_json_comments(raw))
    if isinstance(data, list):
        msgs = data
    elif isinstance(data, dict) and isinstance(data.get("messages"), list):
        msgs = data["messages"]
    else:
        raise ValueError("Template must be a messages array or an object with a 'messages' array.")
    return msgs


def deep_fill(node: Any, vars_map: dict[str, str]) -> Any:
    if isinstance(node, str):
        s = node
        for key, value in vars_map.items():
            s = s.replace(key, value)
        return s
    if isinstance(node, list):
        return [deep_fill(item, vars_map) for item in node]
    if isinstance(node, dict):
        return {k: deep_fill(v, vars_map) for k, v in node.items()}
    return node


def append_extra_images(msgs: list[dict[str, Any]], images: list[dict[str, str]]) -> list[dict[str, Any]]:
    if len(images) <= 1:
        return msgs
    extras = [
        {"type": "image_url", "image_url": {"url": im["data_url"], "detail": "high"}}
        for im in images[1:]
    ]
    out: list[dict[str, Any]] = []
    for msg in msgs:
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            out.append(msg)
            continue
        content = msg["content"]
        if not any(part.get("type") == "image_url" for part in content if isinstance(part, dict)):
            out.append(msg)
            continue
        cloned = deepcopy(msg)
        cloned["content"] = list(cloned["content"]) + extras
        out.append(cloned)
    return out


def image_to_data_url(path: Path) -> dict[str, str]:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"name": path.name, "data_url": f"data:{mime};base64,{data}"}


def pdf_to_images(path: Path, max_pages: int = 20, scale: float = 2.0) -> list[dict[str, str]]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is required for PDFs. Install with: pip install pymupdf")
    doc = fitz.open(path)
    page_count = min(doc.page_count, max_pages)
    base = path.stem
    matrix = fitz.Matrix(scale, scale)
    images: list[dict[str, str]] = []
    for n in range(page_count):
        page = doc.load_page(n)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png = pix.tobytes("png")
        data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        name = f"{base}.png" if page_count == 1 else f"{base}-p{n + 1}.png"
        images.append({"name": name, "data_url": data_url})
    if doc.page_count > max_pages:
        print(
            f"  warning: {path.name} has {doc.page_count} pages; only first {max_pages} were attached.",
            file=sys.stderr,
        )
    return images


def load_attachment(path: Path, max_pdf_pages: int) -> list[dict[str, str]]:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        return pdf_to_images(path, max_pages=max_pdf_pages)
    if ext in IMAGE_EXTS:
        return [image_to_data_url(path)]
    raise ValueError(f"Unsupported file type: {path.name}")


def build_messages(
    template_msgs: list[dict[str, Any]],
    images: list[dict[str, str]],
    categories: str,
) -> list[dict[str, Any]]:
    if not images:
        raise ValueError("No images to attach.")
    vars_map = {
        "{{IMAGE_DATA_URL}}": images[0]["data_url"],
        "{{RECEIPT_CATEGORIES}}": categories,
    }
    filled = deep_fill(template_msgs, vars_map)
    return append_extra_images(filled, images)


def chat_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if re.search(r"/v\d+$", base):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def build_body(
    messages: list[dict[str, Any]],
    *,
    model: str | None,
    stream: bool,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": messages, "stream": stream}
    if model:
        body["model"] = model
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if seed is not None:
        body["seed"] = seed
    return body


def headers(api_key: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def read_stream(response) -> tuple[str, str]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if reasoning:
            reasoning_parts.append(reasoning)
    return "".join(content_parts), "".join(reasoning_parts)


def call_api(
    url: str,
    body: dict[str, Any],
    api_key: str | None,
    timeout: int,
) -> tuple[str, str, dict[str, Any] | None]:
    if requests is None:
        raise RuntimeError("requests is required. Install with: pip install requests")

    try:
        res = requests.post(
            chat_url(url),
            headers=headers(api_key),
            json=body,
            timeout=timeout,
            stream=bool(body.get("stream")),
        )
    except requests.RequestException as exc:
        raise RuntimeError(str(exc)) from exc
    if not res.ok:
        raise RuntimeError(f"HTTP {res.status_code}: {res.text}")

    if body.get("stream"):
        content, reasoning = read_stream(res)
        return content, reasoning, None

    data = res.json()
    msg = ((data.get("choices") or [{}])[0]).get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return content, reasoning, data


def discover_files(
    file_arg: str | None,
    dir_arg: str | None,
    *,
    recursive: bool,
) -> list[Path]:
    if bool(file_arg) == bool(dir_arg):
        raise ValueError("Provide exactly one of --file or --dir.")

    allowed = IMAGE_EXTS | PDF_EXTS
    if file_arg:
        path = Path(file_arg).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in allowed:
            raise ValueError(f"Unsupported file type: {path.name}")
        return [path]

    root = Path(dir_arg).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    iterator = root.rglob("*") if recursive else root.iterdir()
    files = [
        p.resolve()
        for p in iterator
        if p.is_file() and p.suffix.lower() in allowed
    ]
    files.sort(key=lambda p: p.name.lower())
    if not files:
        raise ValueError(f"No images or PDFs found in {root}")
    return files


def output_paths(output_dir: Path, source_file: Path) -> tuple[Path, Path]:
    stem = source_file.stem
    return (
        output_dir / f"{stem}.assistant.txt",
        output_dir / f"{stem}.response.json",
    )


def save_result(
    output_dir: Path,
    source_file: Path,
    content: str,
    reasoning: str,
    full_response: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assistant_path, response_path = output_paths(output_dir, source_file)

    parts: list[str] = []
    if reasoning.strip():
        parts.append("=== reasoning ===\n" + reasoning.strip())
    if content.strip():
        parts.append("=== content ===\n" + content.strip())
    assistant_path.write_text("\n\n".join(parts) if parts else "", encoding="utf-8")

    payload = {
        "source_file": str(source_file),
        "assistant_content": content,
        "assistant_reasoning": reasoning,
        "api_response": full_response,
    }
    response_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def process_file(
    path: Path,
    *,
    template_msgs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    print(f"\n→ {path.name}")
    images = load_attachment(path, args.max_pdf_pages)
    print(f"  attached {len(images)} image(s)")
    messages = build_messages(template_msgs, images, args.categories)

    if args.dry_run:
        preview = deepcopy(messages)
        for msg in preview:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = ((part.get("image_url") or {}).get("url")) or ""
                    if url.startswith("data:"):
                        part["image_url"]["url"] = url[:48] + f"…[+{len(url) - 48} chars]"
        print(json.dumps(build_body(
            messages,
            model=args.model,
            stream=not args.no_stream,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ) | {"messages": preview}, indent=2)[:4000])
        return

    body = build_body(
        messages,
        model=args.model,
        stream=not args.no_stream,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    started = time.time()
    content, reasoning, full_response = call_api(
        args.url,
        body,
        args.api_key,
        args.timeout,
    )
    elapsed = time.time() - started
    print(f"  done in {elapsed:.1f}s ({len(content)} chars)")

    out_dir = Path(args.output).expanduser().resolve()
    save_result(out_dir, path, content, reasoning, full_response)
    assistant_path, response_path = output_paths(out_dir, path)
    print(f"  saved {assistant_path.name}")
    print(f"  saved {response_path.name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a template against image/PDF files via a local chat completions API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "template",
        nargs="?",
        default=None,
        metavar="TEMPLATE",
        help="Template JSON file (object with 'messages' or a messages array)",
    )
    p.add_argument(
        "-t", "--template",
        dest="template_flag",
        default=None,
        metavar="TEMPLATE",
        help="Same as positional TEMPLATE",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Single image or PDF to process")
    src.add_argument("--dir", help="Folder of images/PDFs to process sequentially")
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output folder (default: {DEFAULT_OUTPUT.name}/)",
    )
    p.add_argument("--categories", default=DEFAULT_CATEGORIES, help="Value for {{RECEIPT_CATEGORIES}}")
    p.add_argument("--url", default=os.environ.get("LOCAL_AI_URL", DEFAULT_URL), help="Chat completions URL")
    p.add_argument(
        "--api-key",
        default=None,
        help=f"Bearer API key (default: LOCAL_AI_API_KEY or {DEFAULT_TOKEN_FILE.name})",
    )
    p.add_argument("--model", default=os.environ.get("LOCAL_AI_MODEL"), help="Model name")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-pdf-pages", type=int, default=20, help="Max PDF pages to convert (default: 20)")
    p.add_argument("--timeout", type=int, default=600, help="HTTP timeout in seconds (default: 600)")
    p.add_argument("--recursive", action="store_true", help="With --dir, include subfolders")
    p.add_argument("--no-stream", action="store_true", help="Disable streaming responses")
    p.add_argument("--dry-run", action="store_true", help="Build request only; do not call the API")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if fitz is None:
        print("error: PyMuPDF is required. Install dependencies with:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return 1
    if requests is None and not args.dry_run:
        print("error: requests is required. Install dependencies with:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return 1

    template_raw = args.template_flag or args.template or str(DEFAULT_TEMPLATE)
    template_path = Path(template_raw).expanduser().resolve()
    if not template_path.is_file():
        print(f"error: template not found: {template_path}", file=sys.stderr)
        return 1

    try:
        template_msgs = parse_template(template_path)
        files = discover_files(args.file, args.dir, recursive=args.recursive)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.api_key = load_api_key(args.api_key)

    print(f"template: {template_path}")
    print(f"files: {len(files)}")
    print(f"endpoint: {chat_url(args.url)}")
    print(f"output: {Path(args.output).expanduser().resolve()}")
    if args.api_key:
        print("api key: loaded")
    elif not args.dry_run:
        print("warning: no API key found (set --api-key, LOCAL_AI_API_KEY, or unsloth-local-token.tk)", file=sys.stderr)

    failures = 0
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}]", end="")
        try:
            process_file(path, template_msgs=template_msgs, args=args)
        except (OSError, ValueError, RuntimeError) as exc:
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)

    if failures:
        print(f"\nFinished with {failures} failure(s).", file=sys.stderr)
        return 1
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
