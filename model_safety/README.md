# Protected model loaders

These nodes retain ComfyUI's native loader implementations and output socket types.
Only allowed filenames, model-family type selection and missing-file checks change.

`node/policy.py` is also used by installers to create new versioned protected copies.
It does not rewrite a malformed graph silently. The optional repository repair tool
can correct known single-family loader mistakes into a new file while preserving
prompts and graph links.

The protection does not make differently wired model families interchangeable.
Rewiring a KREA encoder into an H3 conditioning node remains unsupported. Existing
unprotected saved workflows remain unprotected until explicitly repaired or replaced
with one of the new installed copies.
