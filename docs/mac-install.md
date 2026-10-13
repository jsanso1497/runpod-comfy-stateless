# Mac installation without Terminal

Use one route only. Both update the same existing repository. You do not need a new GitHub repository or GitHub CLI.

## Before replacing files

Keep a backup of any uncommitted work outside the repository. Keep private libraries and credentials outside the upload. The provided replacement contains no private LoRA text file or model weights.

Replace source files, not the GitHub repository itself. Do not use the repository Settings > Delete repository control. On a local clone, never delete its `.git` folder. A normal replacement commit retains existing history; it does not remove sensitive data from earlier commits or older Docker images.

If you use a protected default branch, do this on one replacement branch and merge its pull request after all uploads/checks are complete. Keep every upload on that same branch. Do not disable branch protections to install.

## Route A: GitHub Desktop and Finder

1. Install GitHub Desktop for macOS and sign in. In GitHub, open your existing repository and choose **Code > Open with GitHub Desktop**, then **Clone**. An already-cloned repository can simply be selected in Desktop.
2. Open the local repository folder in Finder. Press **Command + Shift + Period** to show hidden items. Move the old working files out of the folder, but leave **`.git`**. You can back up the old files in another folder instead of immediately trashing them.
3. Unzip the replacement archive. Open its `runpod-comfy-stateless` folder, show hidden files, and copy **all contents** into the local repository. Do not copy the enclosing folder itself. The new `Dockerfile` must be at the repository root.
4. Confirm `.github/workflows/build.yml`, `.github/workflows/ci.yml`, `.github/workflows/pages.yml`, `.gitignore`, `.dockerignore`, and `.env.example` are in place. Finder visibility does not make these optional.
5. Return to Desktop. Review all additions, modifications and deletions, enter `Replace setup with task-based HQ workbench`, choose **Commit to [branch]**, and click **Push origin**. If this is a new replacement branch, use **Publish branch**, then create/merge the pull request. Do not click Create repository or Publish repository.

Desktop will commit files that are in the local working folder even when Finder normally hides them. Showing hidden files is necessary when you manually select and copy the replacement's contents.

## Route B: Browser only, no app installation

Use the companion **browser-upload kit**, not the standard replacement ZIP. Unzip it and open **START_HERE.html**. Everything in that kit has a visible filename, so Finder cannot hide required payloads.

1. After clearing the old source files on your chosen branch, open the root of the existing GitHub repository. Do not upload or commit any private source file from the old setup.
2. Choose **Add file > Upload files**. In Finder, open **UPLOAD_1_CORE**, select everything inside it, and drag that selection into GitHub. Review the file paths and commit. Upload the *contents*, not the UPLOAD_1_CORE folder.
3. Repeat with **UPLOAD_2_WORKFLOWS**, then **UPLOAD_3_AUTOMATION**. Return to the repository root before every upload. Each batch is under GitHub's 100-file upload limit. The guide, manifest, batch wrappers and DOTFILES_AS_TEXT folder are not uploaded.
4. In the HTML guide, follow the six hidden-file cards. For each one, use **Copy path**, choose **Add file > Create new file** at the repository root, paste the exact path, use **Copy contents**, paste the file contents, and commit to the same branch. GitHub creates subdirectories from slashes in the filename. Create the CI workflow last, after all other files are in place. If a file already exists, edit it rather than adding a duplicate.
5. Confirm the root contains `Dockerfile`, `README.md`, `src`, `catalog`, `workflows`, `automation`, `site`, and `.github`. There must be no enclosing `runpod-comfy-stateless` or `UPLOAD_...` folder inside the repository. Merge your replacement branch if applicable.

GitHub does not unpack a ZIP into source files when you upload it. Uploading the ZIP itself stores an archive, not an installed repository. No personal access token or API access is needed for this route; only your existing signed-in GitHub session and permission to change this repository.

## Build and launch using the browser

1. In the repository's **Actions** tab, run **Validate repository**. Wait for success before building.
2. Run **Build HQ workspace**, choose a workspace, and use the image reference/digest from its successful summary. The build derives image ownership from the actual repository, not an assumed new repository.
3. Open `site/index.html` from the extracted package in your browser. Confirm the owner/repository field, select your tasks, and copy the RunPod settings. The existing secret names remain **hf_token** and **civit_token**. The required access-password secret is **comfy_password**.
4. For a hosted configurator, choose **Settings > Pages > Source: GitHub Actions**, then run **Publish configurator to Pages** and explicitly check its public-catalog consent box. Availability depends on your GitHub plan and repository visibility. The local configurator does not require Pages.

Docker builds, external downloads, live ComfyUI node imports and GPU inference remain unverified in the delivered source package. Offline checks are not evidence of successful GPU execution.

## Official references

- GitHub Desktop cloning: https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop
- Review, commit and push: https://docs.github.com/en/desktop/making-changes-in-a-branch/committing-and-reviewing-changes-to-your-project-in-github-desktop
- Browser uploads and limits: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
- Create files and subdirectories: https://docs.github.com/en/repositories/working-with-files/managing-files/creating-new-files
