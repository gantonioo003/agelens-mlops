"""Resume a large pinned file using bounded, independently checked HTTP ranges."""

import argparse
import concurrent.futures
import shutil
import time
import urllib.request

from scripts.prepare_assets import ROOT


def fetch_range(url, start, end, total, output):
    expected = end - start + 1
    if output.exists() and output.stat().st_size == expected:
        return output
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "AgeLens-research/0.1",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206 or response.headers.get("Content-Range") != (
                    f"bytes {start}-{end}/{total}"
                ):
                    raise ValueError("Server did not honor the requested byte range")
                written = 0
                with output.open("wb") as target:
                    while chunk := response.read(64 * 1024):
                        written += len(chunk)
                        if written > expected:
                            raise ValueError("Range exceeds expected size")
                        target.write(chunk)
            if written != expected:
                raise ValueError("Truncated byte range")
            return output
        except Exception:
            output.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8)
    parser.add_argument("--chunk-kib", type=int, choices=[256, 512, 1024, 2048], default=2048)
    args = parser.parse_args()
    target = (ROOT / args.output).resolve()
    if not target.is_relative_to(ROOT.resolve()) or not args.url.startswith("https://"):
        raise ValueError("Require HTTPS and an output inside this project")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if target.exists():
        raise ValueError("Target already exists; verify it rather than overwrite it")
    if not 0 <= offset <= args.size:
        raise ValueError("Partial file exceeds declared size")
    parts = ROOT / "runtime/download-parts" / target.name
    parts.mkdir(parents=True, exist_ok=True)
    step = args.chunk_kib * 1024
    known = {}
    for file in parts.iterdir():
        try:
            start, end = map(int, file.name.split("-"))
        except ValueError:
            continue
        if offset <= start <= end < args.size and file.stat().st_size == end - start + 1:
            known[start] = end
    ranges = []
    start = offset
    while start < args.size:
        end = known.get(start)
        if end is None:
            next_known = min((value for value in known if value > start), default=args.size)
            end = min(start + step, next_known, args.size) - 1
        ranges.append((start, end))
        start = end + 1
    completed = offset
    print(f"Resuming {target.name}: {offset}/{args.size} bytes", flush=True)
    logged = offset
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {
            pool.submit(fetch_range, args.url, start, end, args.size, parts / f"{start}-{end}"): end
            - start
            + 1
            for start, end in ranges
        }
        for future in concurrent.futures.as_completed(tasks):
            future.result()
            completed += tasks[future]
            if completed - logged >= 16 * 1024**2 or completed == args.size:
                print(f"{100 * completed / args.size:.1f}% ({completed}/{args.size})", flush=True)
                logged = completed
    for start, end in ranges:
        if (parts / f"{start}-{end}").stat().st_size != end - start + 1:
            raise ValueError("A completed range changed before assembly; refuse truncated output")
    if (partial.stat().st_size if partial.exists() else 0) != offset:
        raise ValueError("Partial output changed during download; refuse concurrent writers")
    with partial.open("ab") as out:
        for start, end in ranges:
            part = parts / f"{start}-{end}"
            with part.open("rb") as source:
                shutil.copyfileobj(source, out)
    assert partial.stat().st_size == args.size
    partial.replace(target)
    for start, end in ranges:
        (parts / f"{start}-{end}").unlink()
    print("Transfer complete; verify the asset hash before use.", flush=True)


if __name__ == "__main__":
    main()
