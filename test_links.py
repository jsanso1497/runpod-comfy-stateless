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
