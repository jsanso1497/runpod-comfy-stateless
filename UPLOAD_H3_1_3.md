# H3 Portrait 1.3: re-upload the existing release files

These 31 source files are byte-for-byte copies of their matching files in
runpod-comfy-stateless-complete-2026-10-05.zip. This is NOT a new code revision.
The focused archive excludes your config/lora_links.txt and shared LoRA tests.

## What the uploaded log actually confirms

The uploaded logs_101172084919.zip records a successful portrait build of commit
52b87313513c2c13891b25829eb9f39ff9cf6131 and publication of digest
sha256:5f010687d6551fccd9a504c2d6be56b25ed8a788bdffcd4dedc6b0f514f877c9.
It runs the older seven-step portrait Dockerfile, 57 portrait tests, and the old
publication summary. Its build never invokes verify_release.py and it does not
publish the h3-portrait-v1.3.0 tag configured in the consolidated release.
This is not the complete 1.3 source/build recipe. The log does not establish what
is on main after that commit, or whether another build has since run.

## Update only these paths

1. In jsanso1497/runpod-comfy-stateless on main, upload h3_portrait/ and tools/
   from this extracted archive into the ROOT of the repository. Replace matching
   files. Do not put an enclosing h3-portrait-1.3-files folder into GitHub.
2. Replace .github/workflows/build-h3-portrait.yml with the complete copy supplied
   at the same path. The separately linked YAML is that exact same file.
3. Verify h3_portrait/VERSION reads 1.3.0 and h3_portrait/node/analysis_prompt.txt
   exists. The portrait Dockerfile must invoke verify_release.py.
4. After ALL files are saved, go to Actions > Build H3 Portrait template >
   Run workflow > main > Run workflow. Do not rerun the 52b8731 job.

If those exact source files already match on main because the complete ZIP was
uploaded after the recorded run, skip the upload and start a NEW workflow on main.

## Required build confirmation

H3 PORTRAIT SOURCE VERIFIED: 1.3.0 | split-model-reference-v3
REPOSITORY PREFLIGHT PASS: portrait

Portrait test suite: Ran 150 tests / OK
The successful final summary is: H3 Portrait 1.3 image published

Do not deploy another Pod for this update until that verified build succeeds.
Keep the image tag, GPU, environment variables, port, credentials, and storage
unchanged. Do not replace config/lora_links.txt or shared_loras/tests/test_links.py.
Only the portrait build is required for portrait-only use.

## Validation performed for this focused archive

- Confirmed all 31 code/config files are identical to the complete 1.3 ZIP.
- Ran the existing repository preflight against that complete source: PASS.
- Ran all 150 portrait tests again: PASS. Ollama calls in these tests are mocked.
- No Docker build, registry publication, live repository read, or GPU generation
  was performed during this review. The archive does not change GitHub itself.
