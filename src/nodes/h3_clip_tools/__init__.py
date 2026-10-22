"""H3 clip preparation UI. File-backed VIDEO in, bounded CPU exports out."""
from __future__ import annotations
import json
from pathlib import Path

from comfy_api.latest import ComfyExtension, io, ui
from workbench.h3_clip_tools import (
    AUDIO_MODES, MAX_CLIPS, MODES, TAIL_POLICIES, export_clips, resolve_video,
)


class WBH3ExportClips(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='WBH3ExportClips',
            display_name='H3 Clips - Extract / Split 124 Frames',
            category='Workbench/H3 Clip Preparation',
            description='Prepare long local videos as exact 124-frame, 24 fps source clips. CPU FFmpeg only; no H3 or Ollama inference. Re-encodes at CRF 16 by default. Original file is unchanged.',
            is_output_node=True,
            inputs=[
                io.Video.Input('video', tooltip='Connect native Load Video. Three-minute and longer files are supported; no full-video image batch is created.'),
                io.Combo.Input('mode', options=list(MODES), default=MODES[0]),
                io.Float.Input('start_seconds', default=0.0, min=0.0, max=2592000.0, step=0.01,
                    tooltip='Beginning of the chunk grid. For a single custom section, enter its exact source start and leave first_chunk=1.'),
                io.Int.Input('first_chunk', default=1, min=1, max=500000,
                    tooltip='1-based chunk number from start_seconds. Chunk N begins at start_seconds + (N-1)*124/24.'),
                io.Int.Input('chunk_count', default=0, min=0, max=MAX_CLIPS,
                    tooltip='Consecutive mode only: 0 = all remaining; otherwise export this many, up to end of source. Maximum 1000 per run.'),
                io.Combo.Input('final_chunk', options=list(TAIL_POLICIES), default=TAIL_POLICIES[0],
                    tooltip='Default skips the short final remainder. Optional hold duplicates the final picture to fill 124 frames and marks the clip as padded.'),
                io.Combo.Input('audio', options=list(AUDIO_MODES), default=AUDIO_MODES[0],
                    tooltip='Keep the corresponding source soundtrack, re-encoded to AAC, or make silent motion references. No audio is invented when the source has none.'),
                io.String.Input('output_prefix', default='H3_clips'),
                io.Boolean.Input('make_loadable', default=True,
                    tooltip='Publish uniquely named h3clip_ files in ComfyUI input storage. Refresh the video list and select one in HMOTION, without re-uploading. Hardlink when possible; copy otherwise.'),
                io.Boolean.Input('make_zip', default=True,
                    tooltip='For two or more clips, also create a downloadable ZIP with a CSV manifest. Requires additional disk space.'),
                io.Int.Input('crf', default=16, min=0, max=30,
                    tooltip='H.264 quality. 16 = high quality, not lossless. 0 = lossless relative to the normalized 8-bit YUV420 frames, with larger files.'),
            ],
            outputs=[io.String.Output('export_report')],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # An explicit queue action exports a new uniquely named run. Never return
        # a cached success referring to files that may have been removed.
        return float('nan')

    @classmethod
    def execute(cls, video, mode, start_seconds, first_chunk, chunk_count, final_chunk,
                audio, output_prefix, make_loadable, make_zip, crf):
        import folder_paths
        import comfy.model_management as mm
        import comfy.utils
        source, trim_start, trim_duration = resolve_video(video, folder_paths.get_input_directory())
        bar = None

        def progress(done, total):
            nonlocal bar
            if bar is None:
                bar = comfy.utils.ProgressBar(total)
            bar.update_absolute(done, total)

        output_root = Path(folder_paths.get_output_directory()).resolve()
        report = export_clips(
            source, output_root, folder_paths.get_input_directory(),
            mode=mode, start_seconds=start_seconds, first_chunk=first_chunk, chunk_count=chunk_count,
            final_chunk=final_chunk, audio=audio, output_prefix=output_prefix,
            make_loadable=make_loadable, make_zip=make_zip, crf=crf,
            input_start=trim_start, input_duration=trim_duration,
            check_cancel=mm.throw_exception_if_processing_interrupted, progress=progress,
        )
        rows = report['clips']
        heading = (f"Saved {len(rows)} clip(s), each exactly 124 frames at 24 fps (5.1666667 seconds).\n"
                   f"Folder: {report['directory']}\n"
                   'To use in HMOTION: refresh the video list, select an h3clip_ file, and set source start_seconds=0; leave window_seconds=5.25.\n'
                   'This prepares clips only. It does not queue H3 generations or stitch generated clips.\n')
        if report.get('zip'):
            heading += 'Batch download: Workbench > Files > ' + report['zip'] + '\n'
        heading += '\n' + json.dumps(report, indent=2, ensure_ascii=False)
        # Browser previews are disk-backed files, not decoded long-video tensors.
        previews = [ui.SavedResult(row['file'], report['directory'], io.FolderType.output) for row in rows[:4]]
        return io.NodeOutput(heading, ui=ui.PreviewVideo(previews))


class H3ClipExtension(ComfyExtension):
    async def get_node_list(self):
        return [WBH3ExportClips]


async def comfy_entrypoint():
    return H3ClipExtension()
