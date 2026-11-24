import io
import json
import tarfile
from pathlib import Path
from typing import ClassVar

import pytest

from scripts.check_environment import validate_protocol
from scripts.download_large import fetch_range
from scripts.prepare_assets import download, extract_images

ROOT = Path(__file__).resolve().parents[1]


def make_archive(path, name, content=b"not-a-real-image", link=False):
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo(name)
        if link:
            item.type = tarfile.SYMTYPE
            item.linkname = "outside"
            archive.addfile(item)
        else:
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))


def test_protocol():
    validate_protocol(json.loads((ROOT / "configs/protocol-v1.json").read_text()))


@pytest.mark.parametrize(
    "name,link",
    [
        ("../escape.jpg", False),
        ("/escape.jpg", False),
        ("escape.jpg", True),
        ("..\\escape.jpg", False),
    ],
)
def test_reject_unsafe_tar(tmp_path, name, link):
    archive = tmp_path / "data.tar.gz"
    make_archive(archive, name, link=link)
    with pytest.raises(ValueError):
        extract_images(archive, tmp_path / "images")


def test_extract_regular_file(tmp_path):
    archive = tmp_path / "data.tar.gz"
    make_archive(archive, "UTKFace/18_0_0_example.jpg")
    assert extract_images(archive, tmp_path / "images") == 1
    assert (tmp_path / "images/18_0_0_example.jpg").read_bytes() == b"not-a-real-image"


def test_reject_wrong_existing_hash(tmp_path):
    (tmp_path / "file").write_bytes(b"data")
    with pytest.raises(ValueError, match="SHA-256"):
        download({"path": "file", "size": 4, "sha256": "incorrect"}, root=tmp_path)


def test_bad_download_does_not_publish_target(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"data")
    with pytest.raises(ValueError, match="SHA-256"):
        download(
            {
                "name": "fixture",
                "path": "result",
                "url": source.as_uri(),
                "size": 4,
                "sha256": "incorrect",
            },
            root=tmp_path,
        )
    assert not (tmp_path / "result").exists()
    assert not (tmp_path / "result.part").exists()


def test_range_requires_exact_response(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers: ClassVar[dict] = {}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"data"))
    monkeypatch.setattr("scripts.download_large.time.sleep", lambda _: None)
    with pytest.raises(ValueError, match="byte range"):
        fetch_range("https://example.test/file", 5, 8, 12, tmp_path / "part")
    assert not (tmp_path / "part").exists()


def test_range_writes_only_requested_bytes(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers: ClassVar[dict] = {"Content-Range": "bytes 5-8/12"}

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(b"data"))
    output = fetch_range("https://example.test/file", 5, 8, 12, tmp_path / "part")
    assert output.read_bytes() == b"data"
