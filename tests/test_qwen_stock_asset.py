"""Ensure the stock prompt encoder downloads without weakening BF16 guarantees."""
from pathlib import Path

import workbench.assets as assets
from workbench.config import read_catalog
from workbench.qwen_stock_catalog import PE_FILE

OFFICIAL_SHA = '32707d01b427e488af252b95c551989aad59f9fec611a694f5db6bde7f0f1f6c'


def stock():
    rows = [row for row in read_catalog('assets.json')['assets']
            if row['id'] == 'qwen-stock-pe-i2i-int8']
    assert len(rows) == 1
    return rows[0]


def test_official_stock_encoder_is_pinned():
    row = stock()
    assert row['repo_id'] == 'Comfy-Org/Qwen-Image-2.1'
    assert row['filename'] == 'text_encoders/' + PE_FILE
    assert row['group'] == 'qwen_stock_pe'
    assert row['sha256'] == OFFICIAL_SHA
    assert assets._standard_asset_high_precision(row) is False


def test_int8_exception_is_restricted_to_one_exact_source():
    row = stock()
    variants = [
        {'id': 'other-int8'}, {'group': 'qwen'}, {'family': 'krea'},
        {'repo_id': 'not-approved/example'}, {'filename': 'text_encoders/other.safetensors'},
        {'destination': 'text_encoders/other.safetensors'}, {'sha256': '0' * 64},
    ]
    for changes in variants:
        assert assets._standard_asset_high_precision({**row, **changes}) is True
    for entry in read_catalog('assets.json')['assets']:
        if entry['id'] != row['id']:
            from workbench.moss_catalog import is_stock_quantized_asset
            assert assets._standard_asset_high_precision(entry) is (not is_stock_quantized_asset(entry))


def test_asset_preparation_uses_exception_only_for_stock_encoder(tmp_path, monkeypatch):
    row = stock()
    regular = next(a for a in read_catalog('assets.json')['assets']
                   if a['id'] == 'qwen-qwen_image_2-1_bf16')
    observed = {}

    class FakeTransport:
        def close(self):
            pass

    monkeypatch.setattr(assets.pd, 'Transport', FakeTransport)
    monkeypatch.setattr(assets, 'read_catalog', lambda name: {'assets': [row, regular]})
    monkeypatch.setattr(assets, 'model_root', lambda: tmp_path / 'models')
    monkeypatch.setattr(assets, 'state_root', lambda: tmp_path / 'workbench')
    monkeypatch.setattr(assets, 'resolve_standard', lambda asset, transport: {'revision': 'mock'})

    def fake_transfer(meta, target, transport, *, high_precision):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'unit-test')
        observed[Path(target).name] = high_precision
        return {'bytes': 9}

    monkeypatch.setattr(assets, 'transfer', fake_transfer)
    monkeypatch.setattr(assets.pd, 'file_hash', lambda path: 'fixture-hash')
    result = assets.prepare_standard(['qwen', 'qwen_stock_pe'])
    assert not result['failures']
    assert observed[PE_FILE] is False
    assert observed['qwen_image_2.1_bf16.safetensors'] is True
