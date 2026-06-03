# Training the perception model on the RTX 4050 laptop (6 GB)

The Mac builds the code; the 4050 laptop runs the fine-tune. Training is small
and **DataLoader-bound** (musdb decodes `.stem.mp4` on the fly), so a 6 GB card
is plenty — the GPU mostly waits on CPU audio decode. The `MUSDBLoader` LRU
cache (`cache_size=128`, holds the whole bundled set) keeps it fed.

## 1. Get the project onto the laptop

Copy the `mastering-agent/` folder over (or `git init` + push/pull). You do
**not** need to copy `.venv/`, `outputs/`, or `runs/`. You **do** want
`checkpoints/Cnn14_mAP=0.431.pth` (312 MB) — or re-download it (step 3).

## 2. Python env with a CUDA build of torch

`requirements.txt` pins the **CPU** torch wheels (for the Mac). On the laptop,
install the CUDA build instead:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

# CUDA 12.1 build of the pinned versions (use cu118 if your driver is older):
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# everything else (skip the torch lines in requirements.txt — already installed):
pip install -r requirements.txt
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You also need **ffmpeg on PATH** (stempeg/musdb decode mp4):
- Windows: `winget install ffmpeg` (or `choco install ffmpeg`)
- Linux: `sudo apt install ffmpeg`

(The Mac-specific `/opt/homebrew/bin` PATH shim in `musdb_loader.py` is a no-op
on Windows/Linux. The `SSL_CERT_FILE=certifi` shim is cross-platform.)

## 3. Pretrained checkpoint (if not copied)

```bash
mkdir -p checkpoints
curl -L -o "checkpoints/Cnn14_mAP=0.431.pth" \
  "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
```

## 4. Smoke test (any device, ~10 s)

```bash
python perception/train.py --smoke
# expect: "SMOKE OK — train + val loop ran end-to-end."
```

## 5. Train

```bash
python perception/train.py --epochs 12 --batch-size 8 --num-workers 4
```

- Backbone frozen for epoch 0 (head only, lr 1e-4), unfrozen from epoch 1
  (backbone lr 1e-5). Change with `--unfreeze-epoch`.
- AMP (fp16) is on automatically for CUDA.
- Best-by-val-MSE checkpoint → `checkpoints/perception_best.pth`.
- TensorBoard: `tensorboard --logdir runs/`

**If you hit CUDA OOM:** drop to `--batch-size 4`, or `--chunk-seconds 5`. If
it's still tight, `--unfreeze-epoch 99` keeps the backbone frozen the whole run
(head-only training — lightest, and overfit-resistant on this small dataset).

**Bigger dataset later:** point at full MUSDB18-HQ with
`--root /path/to/musdb18hq --is-wav --chunk-seconds 10 --chunks-per-track 20
--cache-size 16` (lower cache: full tracks are ~85 MB each).

## 6. Bring the checkpoint back to the Mac

Copy `checkpoints/perception_best.pth` back. Load it for inference/the agent:

```python
from perception.model import load_finetuned_perception
from perception.inference import PerceptionInference
model = load_finetuned_perception("checkpoints/perception_best.pth")
perception = PerceptionInference(model)
```

(`build_perception_model(checkpoint_path=...)` is only for the *pretrained*
527-class Cnn14; use `load_finetuned_perception` for our 8-dim checkpoints.)
