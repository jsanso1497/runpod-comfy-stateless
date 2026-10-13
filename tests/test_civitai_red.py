"""Offline regression tests for the Civitai .com/.red downloader allowlist."""
from urllib.parse import parse_qs, urlsplit

import pytest
from workbench import assets
from workbench import provider_downloads as pd

SHA = 'a' * 64
VERSION = 7315
FILE_ID = 93402
MODEL_ID = 54321


class FakeCivitai:
    def __init__(self, file_host='civitai.com'):
        self.urls = []
        self.file_host = file_host

    def json(self, url):
        self.urls.append(url)
        if url.endswith(f'/api/v1/models/{MODEL_ID}'):
            return {'modelVersions': [
                {'id': VERSION, 'status': 'Published', 'files': [1], 'publishedAt': '2026-01-01'}]}
        if url.endswith(f'/api/v1/model-versions/{VERSION}'):
            return {'id': VERSION, 'modelId': MODEL_ID, 'baseModel': 'Qwen Image 2.1',
                    'model': {'type': 'LORA', 'name': 'Example Dog LoRA'},
                    'files': [{'id': FILE_ID, 'name': 'dog-identity.safetensors',
                               'hashes': {'SHA256': SHA}, 'sizeKB': 1024,
                               'downloadUrl': f'https://{self.file_host}/api/download/models/{VERSION}?type=Model',
                               'primary': True}], 'trainedWords': ['dogtoken']}
        raise AssertionError('Unexpected metadata host or route: ' + url)


def test_red_fileid_download_uses_red_api_and_preserves_file_choice():
    t = FakeCivitai(file_host='civitai.com')
    row = pd.resolve_link(f'https://civitai.red/api/download/models/{VERSION}?fileId={FILE_ID}', t)
    assert t.urls == [f'https://civitai.red/api/v1/model-versions/{VERSION}']
    assert urlsplit(row['download_url']).hostname == 'civitai.red'
    assert parse_qs(urlsplit(row['download_url']).query)['fileId'] == [str(FILE_ID)]
    assert row['file_id'] == FILE_ID
    assert row['sha256'] == SHA


def test_red_model_page_uses_red_for_both_metadata_requests():
    t = FakeCivitai(file_host='civitai.red')
    row = pd.resolve_link(f'https://civitai.red/models/{MODEL_ID}', t)
    assert len(t.urls) == 2
    assert all(urlsplit(u).hostname == 'civitai.red' for u in t.urls)
    assert urlsplit(row['download_url']).hostname == 'civitai.red'


def test_dot_com_still_downloads_dot_com():
    t = FakeCivitai(file_host='civitai.com')
    row = pd.resolve_link(f'https://civitai.com/api/download/models/{VERSION}?fileId={FILE_ID}', t)
    assert urlsplit(row['download_url']).hostname == 'civitai.com'


def test_original_red_link_accepted_in_private_library():
    rows = assets.parse_library(
        f'qwen | https://civitai.red/api/download/models/{VERSION}?fileId={FILE_ID}', 'loras')
    assert rows == [{'url': f'https://civitai.red/api/download/models/{VERSION}?fileId={FILE_ID}',
                     'kind': 'loras', 'family': 'qwen'}]


@pytest.mark.parametrize('url', [
    'https://civitai.red.evil.test/api/download/models/123',
    'http://civitai.red/api/download/models/123',
    'https://civitai.red:444/api/download/models/123',
    'https://bad:c@civitai.red/api/download/models/123',
    'https://civitai.red/api/download/models/123?token=secret',
])
def test_unapproved_or_credential_bearing_urls_rejected(url):
    with pytest.raises(pd.LinkError):
        pd.clean_url(url)


def test_normalize_www_red():
    assert pd.clean_url('https://www.civitai.red/api/download/models/7315') == \
           'https://civitai.red/api/download/models/7315'


def test_wrong_file_id_rejected():
    with pytest.raises(pd.LinkError, match='No matching safetensors'):
        pd.resolve_link(f'https://civitai.red/api/download/models/{VERSION}?fileId=999999', FakeCivitai())


def test_token_only_sent_to_same_civitai_host(monkeypatch):
    monkeypatch.setenv('CIVITAI_TOKEN', 'sentinel-token-do-not-log')
    for host in ('civitai.red', 'civitai.com'):
        first = f'https://{host}/api/download/models/{VERSION}'
        assert pd._headers(first, host)['Authorization'] == 'Bearer sentinel-token-do-not-log'
        for destination in ('cdn.example.com', 'civitai.red' if host == 'civitai.com' else 'civitai.com'):
            redirected = f'https://{destination}/asset'
            assert 'Authorization' not in pd._headers(redirected, host)


def test_metadata_must_match_requested_version():
    class Bad(FakeCivitai):
        def json(self, url):
            data = super().json(url)
            if 'model-versions' in url:
                data['files'][0]['downloadUrl'] = f'https://civitai.red/api/download/models/{VERSION + 1}'
            return data
    with pytest.raises(pd.LinkError, match='matching-version'):
        pd.resolve_link(f'https://civitai.red/api/download/models/{VERSION}?fileId={FILE_ID}', Bad())
