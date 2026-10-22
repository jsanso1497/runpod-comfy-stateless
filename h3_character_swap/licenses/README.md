# Separate upstream licenses

This kit does not include model weights. Model files are downloaded only on the Pod.
Review the applicable upstream licenses before professional or commercial use.

- MiniMax H3 and Akatz Character Swap V1.5: MiniMax H3 Community License Agreement.
  https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA
  https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE
- H3 Relay source: its GPL-3.0 / Apache-2.0 notices remain in the source checkout
  and are copied into this image's licenses folder. Source is retained in the image.
  https://github.com/akatz-ai/h3-relay
- BFS v1.1 has its own character-swap license. The image build retrieves it at
  the same immutable model-repository revision used for the asset lock.
  Read /opt/h3-swap/licenses/BFS_CHARACTER_SWAP.md before enabling its download.
  https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/blob/main/LICENSE_CHARACTER_SWAP.md
- The respective Turbo adapters, SAM, Qwen and base components retain their
  own licenses. A source-code license does not replace the model's license.

H3_SWAP_BFS_LICENSE_REVIEWED=1 is only an explicit preparation switch, not a
legal determination or representation that every use is licensed.
