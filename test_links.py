import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shared_loras_sync',ROOT/'sync.py')
sync=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def weights_bytes():
    header=json.dumps({'lora_unet_x.lora_up.weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    header += b' ' * ((-len(header))%8)
    return struct.pack('<Q',len(header))+header+struct.pack('<f',1.)


def version():
    return {'id':22,'modelId':11,'name':'v1','model':{'type':'LORA','name':'Subject'},
            'baseModel':'MiniMax H3','trainedWords':['subjectA'],
            'files':[{'id':33,'name':'subject_fp32.safetensors','type':'Model','primary':True,
                      'sizeKB':1,'metadata':{'fp':'fp32','format':'SafeTensor','size':'full'},
                      'hashes':{'SHA256':'a'*64},'downloadUrl':'https://civitai.com/api/download/models/22?fp=fp32&format=SafeTensor'}]}


def hf_info(lfs=True,multiple=False):
    item={'rfilename':'weights.safetensors','size':100,'blobId':'b'*40}
    if lfs:item['lfs']={'sha256':'c'*64,'size':100}
    return {'id':'owner/repo','sha':'d'*40,'cardData':{'base_model':'MiniMax-H3'},
            'siblings':[item]+([{'rfilename':'other.safetensors','size':50,'blobId':'e'*40}] if multiple else [])}


class LinkParsing(unittest.TestCase):
    def test_comments_blanks_bom_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'links.txt';p.write_text('\ufeff# hello\n\nhttps://civitai.com/models/11\nhttps://civitai.com/models/11\n')
            self.assertEqual(sync.read_links(p),[(3,'https://civitai.com/models/11')])
    def test_www_and_fragment(self):
        self.assertEqual(sync.clean_url('https://www.civitai.com/models/11/#abc'),'https://civitai.com/models/11')
    def test_angle_brackets(self):
        self.assertEqual(sync.clean_url('<https://civitai.com/models/11>'),'https://civitai.com/models/11')
    def test_no_http(self):
        with self.assertRaises(sync.LinkError):sync.clean_url('http://civitai.com/models/11')
    def test_no_unknown_domain(self):
        with self.assertRaises(sync.LinkError):sync.clean_url('https://evil.test/model.safetensors')
    def test_no_username(self):
        with self.assertRaises(sync.LinkError):sync.clean_url('https://user:pass@civitai.com/models/11')
    def test_no_api_key_query_or_leak(self):
        for name in ('token','api_key','access_token','Authorization'):
            with self.subTest(name=name):
                with self.assertRaises(sync.LinkError) as cm:sync.clean_url(f'https://civitai.com/models/11?{name}=secret123')
                self.assertNotIn('secret123',str(cm.exception))
    def test_invalid_port(self):
        with self.assertRaises(sync.LinkError):sync.clean_url('https://civitai.com:444/models/11')
    def test_hf_blob(self):
        self.assertEqual(sync.parse_hf_url('https://huggingface.co/o/r/blob/main/sub/a%20b.safetensors'),('o/r','main','sub/a b.safetensors'))
    def test_hf_resolve(self):
        self.assertEqual(sync.parse_hf_url('https://huggingface.co/o/r/resolve/main/a.safetensors'),('o/r','main','a.safetensors'))
    def test_hf_repo(self):
        self.assertEqual(sync.parse_hf_url('https://huggingface.co/o/r'),('o/r','main',None))
    def test_hf_tree(self):
        self.assertEqual(sync.parse_hf_url('https://huggingface.co/o/r/tree/dev'),('o/r','dev',None))
    def test_hf_nested_branch(self):
        self.assertEqual(sync.parse_hf_url('https://huggingface.co/o/r/blob/refs%2Fpr%2F2/a.safetensors')[1],'refs/pr/2')
    def test_hf_non_safetensors_rejected(self):
        with self.assertRaises(sync.LinkError):sync.parse_hf_url('https://huggingface.co/o/r/blob/main/model.bin')
    def test_hf_folder_rejected(self):
        with self.assertRaises(sync.LinkError):sync.parse_hf_url('https://huggingface.co/o/r/tree/main/sub')
    def test_hf_path_traversal_rejected(self):
        with self.assertRaises(sync.LinkError):sync.parse_hf_url('https://huggingface.co/o/r/blob/main/../a.safetensors')
    def test_missing_list_explicit_error(self):
        with self.assertRaises(sync.LinkError):sync.read_links('/not/a/real/file.txt')
    def test_bad_line_does_not_discard_good_links(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'links';p.write_text('bad\nhttps://civitai.com/models/11\n');errors=[]
            self.assertEqual(len(sync.read_links(p,errors)),1);self.assertEqual(errors[0]['line'],1)
    def test_check_mode_is_strict_and_reports_line(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'links';p.write_text('# x\nbad\n')
            with self.assertRaisesRegex(sync.LinkError,'line 2'):sync.read_links(p)


class CivitaiResolution(unittest.TestCase):
    def fake(self,data=None):
        t=MagicMock();t.json.return_value=data or version();return t
    def test_version_page_extracts_id(self):
        t=self.fake();r=sync.resolve_civitai('https://civitai.com/models/11/name?modelVersionId=22',t)
        self.assertEqual((r['version_id'],r['file_id']),(22,33))
        self.assertEqual(r['sha256'],'a'*64);self.assertEqual(r['trigger_words'],'subjectA')
    def test_direct_download(self):
        r=sync.resolve_civitai('https://civitai.com/api/download/models/22',self.fake())
        self.assertIn('civitai_22_33',r['lora_name'])
    def test_main_page_newest_version(self):
        t=self.fake();t.json.side_effect=[{'modelVersions':[
            {'id':19,'publishedAt':'2026-01-01','files':[1]},
            {'id':22,'publishedAt':'2026-02-01','files':[1]}]},version()]
        self.assertEqual(sync.resolve_civitai('https://civitai.com/models/11/name',t)['version_id'],22)
    def test_mismatched_page_model(self):
        with self.assertRaises(sync.LinkError):sync.resolve_civitai('https://civitai.com/models/99?modelVersionId=22',self.fake())
    def test_wrong_version(self):
        with self.assertRaises(sync.LinkError):sync.resolve_civitai('https://civitai.com/api/download/models/99',self.fake())
    def test_type_checkpoint_rejected(self):
        d=version();d['model']['type']='Checkpoint'
        with self.assertRaises(sync.LinkError):sync.resolve_civitai('https://civitai.com/api/download/models/22',self.fake(d))
    def test_fp32_download_selection(self):
        d=version();second=dict(d['files'][0],id=44,primary=False,metadata={'fp':'fp16'},downloadUrl='https://civitai.com/api/download/models/22?fp=fp16')
        d['files'].append(second)
        r=sync.resolve_civitai('https://civitai.com/api/download/models/22?fp=fp16',self.fake(d))
        self.assertEqual(r['file_id'],44)
    def test_requested_variant_not_available(self):
        with self.assertRaises(sync.LinkError):sync.resolve_civitai('https://civitai.com/api/download/models/22?fp=fp16',self.fake())
    def test_missing_hash_rejected(self):
        d=version();d['files'][0]['hashes']={}
        with self.assertRaises(sync.LinkError):sync.resolve_civitai('https://civitai.com/api/download/models/22',self.fake(d))
    def test_ambiguous_files_requires_link_not_numeric_typing(self):
        d=version();d['files'][0]['primary']=False;d['files'].append(dict(d['files'][0],id=34))
        with self.assertRaisesRegex(sync.LinkError,'Download link'):sync.resolve_civitai('https://civitai.com/api/download/models/22',self.fake(d))
    def test_file_id_from_existing_download_link(self):
        d=version();d['files'].append(dict(d['files'][0],id=44,primary=False))
        self.assertEqual(sync.resolve_civitai('https://civitai.com/api/download/models/22?fileId=44',self.fake(d))['file_id'],44)
    def test_cdn_in_metadata_rejected(self):
        d=version();d['files'][0]['downloadUrl']='https://evil.test/file'
        with self.assertRaises(sync.LinkError):sync.resolve_civitai('https://civitai.com/api/download/models/22',self.fake(d))
    def test_nonfinite_size_rejected(self):
        d=version();d['files'][0]['sizeKB']=float('inf')
        with self.assertRaises(sync.LinkError):sync.resolve_civitai('https://civitai.com/api/download/models/22',self.fake(d))


class HFResolution(unittest.TestCase):
    def fake(self,data):
        t=MagicMock();t.json.return_value=data;return t
    def test_blob_url_becomes_immutable_download(self):
        r=sync.resolve_hf('https://huggingface.co/owner/repo/blob/main/weights.safetensors',self.fake(hf_info()))
        self.assertEqual(r['sha256'],'c'*64);self.assertIn('/resolve/'+'d'*40+'/',r['download_url'])
    def test_repo_unique_file(self):
        r=sync.resolve_hf('https://huggingface.co/owner/repo',self.fake(hf_info()))
        self.assertEqual(r['name'],'weights.safetensors')
    def test_repo_multiple_files_is_not_guessed(self):
        with self.assertRaisesRegex(sync.LinkError,'Open the desired LoRA file'):
            sync.resolve_hf('https://huggingface.co/owner/repo',self.fake(hf_info(multiple=True)))
    def test_exact_file_in_multi_repo(self):
        r=sync.resolve_hf('https://huggingface.co/owner/repo/blob/main/weights.safetensors',self.fake(hf_info(multiple=True)))
        self.assertEqual(r['sha256'],'c'*64)
    def test_git_blob_checksum_not_confused_with_sha256(self):
        r=sync.resolve_hf('https://huggingface.co/owner/repo',self.fake(hf_info(lfs=False)))
        self.assertEqual(r['sha256'],'');self.assertEqual(r['git_blob_sha1'],'b'*40)
    def test_missing_hash_rejected(self):
        d=hf_info(False);d['siblings'][0].pop('blobId')
        with self.assertRaises(sync.LinkError):sync.resolve_hf('https://huggingface.co/owner/repo',self.fake(d))
    def test_missing_file(self):
        with self.assertRaises(sync.LinkError):sync.resolve_hf('https://huggingface.co/owner/repo/blob/main/absent.safetensors',self.fake(hf_info()))
    def test_repo_revision_is_quoted(self):
        t=self.fake(hf_info());sync.resolve_hf('https://huggingface.co/owner/repo/blob/refs%2Fpr%2F2/weights.safetensors',t)
        self.assertIn('/revision/refs%2Fpr%2F2?',t.json.call_args.args[0])


class SecurityAndFiles(unittest.TestCase):
    def test_tokens_only_on_their_origin(self):
        with patch.dict(os.environ,{'HF_TOKEN':'hf_secret','CIVITAI_TOKEN':'civi_secret'}):
            self.assertEqual(sync._headers('https://huggingface.co/a','huggingface.co')['Authorization'],'Bearer hf_secret')
            self.assertEqual(sync._headers('https://civitai.com/a','civitai.com')['Authorization'],'Bearer civi_secret')
            self.assertNotIn('Authorization',sync._headers('https://cdn.test/a','huggingface.co'))
            self.assertNotIn('Authorization',sync._headers('https://civitai.com/a','huggingface.co'))
    def test_unresolved_secret_message(self):
        with patch.dict(os.environ,{'HF_TOKEN':'{{ SECRET }}'}):
            with self.assertRaises(sync.LinkError):sync._headers('https://huggingface.co/a','huggingface.co')
    def test_private_redirect_rejected(self):
        for ip in ('127.0.0.1','10.0.0.1','169.254.169.254','::1'):
            with self.subTest(ip=ip),patch.object(sync.socket,'getaddrinfo',return_value=[(0,0,0,'',(ip,443))]):
                with self.assertRaises(sync.LinkError):sync._check_public_https('https://example.test/file')
    def test_cleartext_redirect_rejected(self):
        with self.assertRaises(sync.LinkError):sync._check_public_https('http://cdn.test/file')
    def test_stream_redirect_strips_token(self):
        with patch.dict(os.environ,{'HF_TOKEN':'secret'}),patch.object(sync,'_check_public_https'):
            t=sync.Transport();a=MagicMock(status_code=302,headers={'Location':'https://cdn.test/file'});b=MagicMock(status_code=200)
            t.session.request=MagicMock(side_effect=[a,b]);t.request('GET','https://huggingface.co/a',redirects=True)
            calls=t.session.request.call_args_list
            self.assertIn('Authorization',calls[0].kwargs['headers']);self.assertNotIn('Authorization',calls[1].kwargs['headers']);t.close()
    def test_no_environment_proxy(self):
        t=sync.Transport();self.assertFalse(t.session.trust_env);t.close()
    def test_git_hash_matches_git_object_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'f';p.write_bytes(b'abc')
            self.assertEqual(sync.file_hash(p,'git'),hashlib.sha1(b'blob 3\0abc').hexdigest())
    def test_structurally_valid_small_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';p.write_bytes(weights_bytes());sync.check_adapter_file(p)
    def test_html_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';p.write_text('<html>failure</html>')
            with self.assertRaises(sync.LinkError):sync.check_adapter_file(p)
    def test_checkpoint_keys_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a';header=json.dumps({'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode();p.write_bytes(struct.pack('<Q',len(header))+header+b'0000')
            with self.assertRaises(sync.LinkError):sync.check_adapter_file(p)
    def test_destination_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(sync.LinkError):sync._destination(Path(td),'../../../file.safetensors')
    def test_symlink_destination_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'models/loras').mkdir(parents=True);(p/'other').write_text('x');(p/'models/loras/a').symlink_to(p/'other')
            with self.assertRaises(sync.LinkError):sync._destination(p,'a')
    def test_atomic_index_no_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a').write_text('x');(p/'index').symlink_to(p/'a')
            with self.assertRaises(sync.LinkError):sync.atomic_json(p/'index',{})
    def test_http_auth_error_is_actionable(self):
        for code in (401,403):
            with self.assertRaisesRegex(sync.LinkError,'token'):sync.status_error(code)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        # These are tiny, mocked downloads. They must not require 8 GiB of
        # free space on the CI host just to exercise the production downloader.
        # Test the real reserve checks below with explicit disk-size fixtures.
        disk_patch = patch.object(
            sync.shutil,
            'disk_usage',
            return_value=SimpleNamespace(
                total=512 * 1024**3, used=0, free=512 * 1024**3,
            ),
        )
        self.disk_usage = disk_patch.start()
        self.addCleanup(disk_patch.stop)

    def row(self,payload):
        return {'lora_name':'Shared/test.safetensors','sha256':hashlib.sha256(payload).hexdigest(),
                'estimated_bytes':len(payload),'download_url':'https://civitai.com/api/download/models/22'}
    def response(self,payload,code=200,headers=None):
        r=MagicMock(status_code=code,headers=headers or {});r.__enter__.return_value=r;r.iter_content.return_value=iter([payload]);return r
    @patch.object(sync.time,'sleep')
    def test_verified_download(self,_):
        with tempfile.TemporaryDirectory() as td:
            payload=weights_bytes();t=MagicMock();t.request.return_value=self.response(payload)
            p=sync.download(self.row(payload),td,t);self.assertEqual(p.read_bytes(),payload)
    def test_verified_cache_no_network(self):
        with tempfile.TemporaryDirectory() as td:
            payload=weights_bytes();row=self.row(payload);p=Path(td)/'models/loras'/row['lora_name'];p.parent.mkdir(parents=True);p.write_bytes(payload)
            t=MagicMock();sync.download(row,td,t);t.request.assert_not_called()
    @patch.object(sync.time,'sleep')
    def test_corrupt_hash_never_published(self,_):
        with tempfile.TemporaryDirectory() as td:
            payload=weights_bytes();row=self.row(payload);t=MagicMock();t.request.side_effect=[self.response(b'wrong') for _ in range(3)]
            with self.assertRaises(sync.LinkError):sync.download(row,td,t)
            self.assertFalse((Path(td)/'models/loras'/row['lora_name']).exists())
    def test_low_disk_prevents_download(self):
        with tempfile.TemporaryDirectory() as td,patch.object(sync.shutil,'disk_usage',return_value=type('D',(),{'free':1})()):
            t=MagicMock()
            with self.assertRaisesRegex(sync.LinkError,'temporary disk'):sync.download(self.row(weights_bytes()),td,t)
            t.request.assert_not_called()
    def test_resume_partial_valid_range(self):
        with tempfile.TemporaryDirectory() as td:
            payload=weights_bytes();row=self.row(payload);p=Path(td)/'models/loras'/row['lora_name'];p.parent.mkdir(parents=True);p.with_suffix('.safetensors.part').write_bytes(payload[:8])
            t=MagicMock();t.request.return_value=self.response(payload[8:],206,{'Content-Range':f'bytes 8-{len(payload)-1}/{len(payload)}'})
            self.assertEqual(sync.download(row,td,t).read_bytes(),payload)
            self.assertEqual(t.request.call_args.kwargs['headers']['Range'],'bytes=8-')
    def test_server_ignores_range_restarts(self):
        with tempfile.TemporaryDirectory() as td:
            payload=weights_bytes();row=self.row(payload);p=Path(td)/'models/loras'/row['lora_name'];p.parent.mkdir(parents=True);p.with_suffix('.safetensors.part').write_bytes(payload[:8])
            t=MagicMock();t.request.return_value=self.response(payload)
            self.assertEqual(sync.download(row,td,t).read_bytes(),payload)
    def test_production_reserve_remains_eight_gib(self):
        self.assertEqual(sync.RESERVE, 8 * 1024**3)

    def test_one_byte_below_required_space_blocks_before_request(self):
        with tempfile.TemporaryDirectory() as td:
            payload = weights_bytes()
            self.disk_usage.return_value = SimpleNamespace(
                free=sync.RESERVE + len(payload) - 1,
            )
            transport = MagicMock()
            with self.assertRaisesRegex(sync.LinkError, 'temporary disk'):
                sync.download(self.row(payload), td, transport)
            transport.request.assert_not_called()

    def test_exact_file_size_plus_reserve_allows_download(self):
        with tempfile.TemporaryDirectory() as td:
            payload = weights_bytes()
            self.disk_usage.return_value = SimpleNamespace(
                free=sync.RESERVE + len(payload),
            )
            transport = MagicMock()
            transport.request.return_value = self.response(payload)
            output = sync.download(self.row(payload), td, transport)
            self.assertEqual(output.read_bytes(), payload)
            self.assertGreaterEqual(self.disk_usage.call_count, 3)

    def test_verified_cache_does_not_need_new_disk_space(self):
        with tempfile.TemporaryDirectory() as td:
            payload = weights_bytes()
            row = self.row(payload)
            target = Path(td) / 'models/loras' / row['lora_name']
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
            self.disk_usage.return_value = SimpleNamespace(free=0)
            transport = MagicMock()
            self.assertEqual(sync.download(row, td, transport), target)
            transport.request.assert_not_called()
            self.disk_usage.assert_not_called()

    def test_resume_reserves_only_remaining_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            payload = weights_bytes()
            row = self.row(payload)
            target = Path(td) / 'models/loras' / row['lora_name']
            target.parent.mkdir(parents=True)
            partial = target.with_suffix('.safetensors.part')
            partial.write_bytes(payload[:8])
            self.disk_usage.return_value = SimpleNamespace(
                free=sync.RESERVE + len(payload) - 8,
            )
            transport = MagicMock()
            transport.request.return_value = self.response(
                payload[8:], 206,
                {'Content-Range': f'bytes 8-{len(payload)-1}/{len(payload)}'},
            )
            self.assertEqual(sync.download(row, td, transport).read_bytes(), payload)
            self.assertEqual(transport.request.call_args.kwargs['headers']['Range'], 'bytes=8-')
            self.assertFalse(partial.exists())

    @patch.object(sync.time, 'sleep')
    def test_ignored_range_rechecks_space_for_full_file(self, _):
        with tempfile.TemporaryDirectory() as td:
            payload = weights_bytes()
            row = self.row(payload)
            target = Path(td) / 'models/loras' / row['lora_name']
            target.parent.mkdir(parents=True)
            partial = target.with_suffix('.safetensors.part')
            partial.write_bytes(payload[:8])
            # Enough for a resumed transfer, but not for a complete restart.
            self.disk_usage.return_value = SimpleNamespace(
                free=sync.RESERVE + len(payload) - 8,
            )
            responses = [self.response(payload) for _attempt in range(3)]
            transport = MagicMock()
            transport.request.side_effect = responses
            with self.assertRaisesRegex(sync.LinkError, 'insufficient temporary disk headroom'):
                sync.download(row, td, transport)
            self.assertEqual(transport.request.call_count, 3)
            for response in responses:
                response.iter_content.assert_not_called()
            self.assertFalse(target.exists())
            self.assertEqual(partial.read_bytes(), payload[:8])

    @patch.object(sync.time, 'sleep')
    def test_disk_drop_interrupts_transfer_without_publishing(self, _):
        with tempfile.TemporaryDirectory() as td:
            payload = weights_bytes()
            row = self.row(payload)
            enough = SimpleNamespace(free=sync.RESERVE + len(payload))
            low = SimpleNamespace(free=sync.RESERVE - 1)
            self.disk_usage.side_effect = [enough, enough, low, low]
            response = self.response(payload)
            response.iter_content.return_value = iter([payload[:8], payload[8:]])
            transport = MagicMock()
            transport.request.return_value = response
            with self.assertRaisesRegex(sync.LinkError, 'temporary disk'):
                sync.download(row, td, transport)
            target = Path(td) / 'models/loras' / row['lora_name']
            self.assertEqual(transport.request.call_count, 1)
            self.assertFalse(target.exists())
            self.assertEqual(target.with_suffix('.safetensors.part').read_bytes(), payload[:8])

    def test_empty_library_index_created(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'links.txt';p.write_text('# empty\n');rows,failed=sync.sync_library(p,td,MagicMock())
            self.assertEqual((rows,failed),([],[]));self.assertTrue((Path(td)/'models/loras/link-library.json').is_file())
    def test_download_failure_does_not_block_other_links(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'links';p.write_text('https://civitai.com/models/11\nhttps://civitai.com/models/12\n')
            payload=weights_bytes();q=Path(td)/'ok.safetensors';q.write_bytes(payload);row=self.row(payload);row.update(name='ok',trigger_words='')
            with patch.object(sync,'resolve_link',side_effect=[sync.LinkError('denied'),row]),patch.object(sync,'download',return_value=q):
                rows,failed=sync.sync_library(p,td,MagicMock())
            self.assertEqual(len(rows),1);self.assertEqual(len(failed),1);self.assertNotIn('download_url',rows[0])

if __name__=='__main__':unittest.main()
