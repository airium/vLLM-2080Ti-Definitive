# SPDX-License-Identifier: Apache-2.0

import os

from vllm.v1.kv_offload.tiering.fs.cache_index import FileSystemCacheIndex


def _block_path(root, hash_hex: str, group: int = 0, rank: int = 0):
    path = root / f"model_abc_r{rank}" / hash_hex[:3] / f"{hash_hex[3:5]}_g{group}"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{hash_hex}.bin"


def _write(path, size: int = 8):
    path.write_bytes(b"x" * size)


def test_reconcile_groups_ranks_and_eviction_order(tmp_path):
    first_rank = _block_path(tmp_path, "a" * 64, group=0, rank=0)
    second_rank = _block_path(tmp_path, "a" * 64, group=1, rank=1)
    protected = _block_path(tmp_path, "b" * 64)
    probationary = _block_path(tmp_path, "c" * 64)
    for path in (first_rank, second_rank, protected, probationary):
        _write(path)

    index = FileSystemCacheIndex(str(tmp_path))
    try:
        index.reconcile()
        assert index.total_bytes() == 32
        index.mark_access([protected])
        removed_groups, removed_bytes = index.cleanup(
            max_disk_bytes=31,
            target_disk_bytes=8,
            min_disk_free_bytes=0,
        )
        assert removed_groups == 2
        assert removed_bytes == 24
        assert not first_rank.exists()
        assert not second_rank.exists()
        assert not probationary.exists()
        assert protected.exists()
    finally:
        index.close()


def test_active_lease_prevents_group_eviction(tmp_path):
    path = _block_path(tmp_path, "d" * 64)
    _write(path)
    index = FileSystemCacheIndex(str(tmp_path))
    try:
        index.reconcile()
        index.reserve([str(path)], "request", ttl_ns=60_000_000_000)
        assert index.cleanup(7, 1, 0) == (0, 0)
        assert path.exists()
        index.release("request")
        assert index.cleanup(7, 1, 0) == (1, 8)
        assert not path.exists()
    finally:
        index.close()


def test_expired_lease_is_reclaimable(tmp_path):
    path = _block_path(tmp_path, "e" * 64)
    _write(path)
    index = FileSystemCacheIndex(str(tmp_path))
    try:
        index.reconcile()
        index.reserve([str(path)], "expired", ttl_ns=-1)
        assert index.cleanup(7, 1, 0) == (1, 8)
        assert not path.exists()
    finally:
        index.close()


def test_reconcile_removes_manually_deleted_entries(tmp_path):
    path = _block_path(tmp_path, "f" * 64)
    _write(path)
    index = FileSystemCacheIndex(str(tmp_path))
    try:
        index.reconcile()
        path.unlink()
        index.reconcile()
        assert index.total_bytes() == 0
    finally:
        index.close()


def test_cleanup_is_disabled_without_thresholds(tmp_path):
    path = _block_path(tmp_path, "1" * 64)
    _write(path)
    index = FileSystemCacheIndex(str(tmp_path))
    try:
        index.reconcile()
        assert index.cleanup(0, 0, 0) == (0, 0)
        assert path.exists()
        assert os.path.exists(index.db_path)
    finally:
        index.close()


def test_quota_does_not_evict_before_high_watermark(tmp_path):
    path = _block_path(tmp_path, "2" * 64)
    _write(path)
    index = FileSystemCacheIndex(str(tmp_path))
    try:
        index.reconcile()
        assert index.cleanup(16, 1, 0) == (0, 0)
        assert path.exists()
    finally:
        index.close()


def test_minimum_free_space_can_trigger_cleanup(tmp_path, monkeypatch):
    first = _block_path(tmp_path, "3" * 64)
    second = _block_path(tmp_path, "4" * 64)
    _write(first)
    _write(second)
    index = FileSystemCacheIndex(str(tmp_path))
    try:
        index.reconcile()
        monkeypatch.setattr(
            index,
            "_free_bytes",
            lambda: 0 if first.exists() and second.exists() else 100,
        )
        removed_groups, removed_bytes = index.cleanup(0, 0, 100)
        assert (removed_groups, removed_bytes) == (1, 8)
        assert first.exists() != second.exists()
    finally:
        index.close()
