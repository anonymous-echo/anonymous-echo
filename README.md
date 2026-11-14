# Echo: Coupled Action and Task Status Signal Denoising

Anonymous code for **Echo: Coupled Action and Task Status Signal Denoising for
Vision Language Action Models**. Echo denoises the robot action together with a
Stop Signal (SS) and a Failure Signal (FS). It is implemented on π0.5 (PI05) and
GR00T N1.5 and evaluated on LIBERO-Recovery. A run needs one Linux machine with
8 NVIDIA GPUs.

## 1. Environment

Run every command from the repository root. PI05 and GR00T use separate Python
3.10 environments, because their torch and transformers versions conflict.

```bash
sudo apt-get install -y ffmpeg git-lfs libegl1 libgl1 libosmesa6
git lfs install

# PI05 and the simulator
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-echo.txt
python -m pip install --no-deps -e LIBERO
deactivate

# GR00T
python3.10 -m venv .venv-gr00t
source .venv-gr00t/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements-gr00t.txt
export PYTHONPATH="$PWD"
python -m echo.models.gr00t.fetch --upstream vendor/Isaac-GR00T   # pinned source + weights
python - <<'PY'
import hashlib, urllib.request
from pathlib import Path
name = "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
target = Path("data/wheels") / name
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    urllib.request.urlretrieve(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/" + name, target)
assert hashlib.sha256(target.read_bytes()).hexdigest() == \
    "b267f80a08e516292cdd748056a2178a45b8abedf7fca123292eb17c21c8c87c", "wheel checksum differs"
PY
python -m pip install --no-deps data/wheels/flash_attn-*.whl
deactivate
```

Before every PI05 command run:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/LIBERO" MUJOCO_GL=egl WANDB_MODE=disabled
```

The entry points check the pinned Python and package versions and stop on a
mismatch. Hugging Face and GitHub downloads may need a proxy. Set it only on the
download command (`http_proxy=... https_proxy=... python -m ...`) and not for
the whole shell.

## 2. Data

| Data | Source | Size |
| --- | --- | ---: |
| Success demonstrations | [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero), [yzembodied/libero_90_image](https://huggingface.co/datasets/yzembodied/libero_90_image) | 6,193 trajectories |
| Recovery demonstrations | [LIBERO_Recovery_Expert](https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_Expert) | 3,184 trajectories |
| Failure rollouts | [anonymous-echo/anonymous-echo](https://huggingface.co/datasets/anonymous-echo/anonymous-echo) | 9,521 trajectories |
| Evaluation scenes | [LIBERO_Recovery_FailureScenes](https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_FailureScenes) | 4,803 scenes |
| PI05 initial weights | [tenstep/pi_model_starvla](https://huggingface.co/tenstep/pi_model_starvla) | |
| GR00T N1.5 weights | [nvidia/GR00T-N1.5-3B](https://huggingface.co/nvidia/GR00T-N1.5-3B) | |

All revisions are pinned in `configs/data.json`. `--source failure` downloads
the 96 failure shards, checks every SHA-256 against `archives.json` and
`inventory.json`, and extracts them to `data/failure`.

```bash
# PI05 environment
python -m echo.data.fetch --source all
python -m echo.data.current_scenes            # evaluation scenes, checks their hashes
```

`python -m echo.data.validation --training-config <config>` reads every training
file once and writes the receipt named in that config. Training refuses to start
without a matching receipt.

## 3. Experiments

`configs/experiments.json` lists every experiment. Each training config is
`configs/<key>.json` and each evaluation config is `configs/eval_<key>.json`.

| Key | Backbone | Description |
| --- | --- | --- |
| `pi05_baseline` | PI05 | Success + recovery data, actions only |
| `pi05_echo07` | PI05 | Echo, τ<sub>clean</sub> = 0.7 |
| `pi05_boundary04` | PI05 | Echo 0.7, success end points shifted by up to ±0.2 s |
| `pi05_classifier3` | PI05 | Three-class status classifier, tail-balanced sampling |
| `pi05_sequential3` | PI05 | Three-class status head feeding the action head, tail-balanced sampling |
| `gr00t_baseline` | GR00T | Success + recovery data, actions only |
| `gr00t_baseline_failure` | GR00T | Baseline that also samples failure chunks, with zero action loss on them |
| `gr00t_echo07` | GR00T | Echo, τ<sub>clean</sub> = 0.7 |
| `gr00t_boundary04` | GR00T | Echo 0.7 with the same ±0.2 s end-point shift |
| `gr00t_signal_isolated` | GR00T | Signals predicted by a separate network that shares no trainable weights with the action network |

All experiments except `pi05_init` and `gr00t_init` train 2,500 steps with
global batch 128 on 8 GPUs (PI05: 16 per GPU; GR00T: micro-batch 4, 4
accumulation steps). They start from a shared success + recovery model trained
for 10,000 steps, which you train first.

### Train

```bash
# PI05 (.venv): the init run first, then any experiment.
for KEY in pi05_init pi05_echo07; do
  python -m echo.data.validation --training-config configs/$KEY.json
  python -m echo.training.train --config configs/$KEY.json --preflight
  torchrun --standalone --nproc_per_node=8 -m echo.training.train --config configs/$KEY.json
done

# GR00T (.venv-gr00t)
for KEY in gr00t_init gr00t_echo07; do
  python -m echo.data.validation --training-config configs/$KEY.json
  python -m echo.models.gr00t.train --config configs/$KEY.json --preflight
  torchrun --standalone --nproc_per_node=8 -m echo.models.gr00t.train --config configs/$KEY.json
done
```

Replace `pi05_echo07` or `gr00t_echo07` with any other key of the same backbone.
The init run needs to be trained only once per backbone. Validate PI05
experiments after the init run, because they read its normalization statistics. Checkpoints go
to `runs/<key>/checkpoints/step_00002500`. To continue an interrupted run, add
`--resume runs/<key>/checkpoints/step_<last>`.

The init run is trained here, so numbers differ from runs that start from other
weights. Training on 8 GPUs also changes the batch partition, so a run is not
bitwise identical to one on a different GPU count.

## 4. Evaluation

Every experiment is evaluated on all 48,030 trials (4,803 scenes × 10). One
trial runs 520 actions, executes 8 actions per query and counts a success at its
first occurrence. Use `.venv` for the simulator in both backbones. GR00T
starts its own worker in `.venv-gr00t` (set in the eval config).

```bash
KEY=pi05_echo07                        # any key with an evaluation config
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i MUJOCO_EGL_DEVICE_ID=$i EGL_DEVICE_ID=$i \
    python -m echo.evaluation.evaluate --config configs/eval_$KEY.json \
      --num-shards 8 --shard-index $i --render-gpu-id $i &
done; wait
python -m echo.evaluation.results summary --run runs/eval_$KEY
```

Rerunning the same command continues an interrupted evaluation. The shard count
is fixed by the first launch. `results summary` writes `runs/eval_$KEY/summary.json`
with the success count and rate, overall and per suite. Add
`--baseline runs/eval_<baseline key>` to also report the baseline on the same
trials.

Echo, boundary and status runs also record SS and FS. Score them with:

```bash
python -m echo.evaluation.signals --run runs/eval_$KEY --config configs/signals.json
```

Compare a run with its baseline on the same trials (both runs must be complete):

```bash
python -m echo.evaluation.compare --run runs/eval_pi05_echo07 --baseline runs/eval_pi05_baseline
```

The trials of one scene are correlated, so the test treats the 4,803 scenes as
the units. It reports the success difference, a 95% scene-bootstrap interval
(4,000 resamples), a paired sign-flip p-value (10<sup>6</sup> random flips,
two-sided) and a normal-approximation p-value on the scene totals. The result
goes to `runs/eval_pi05_echo07/comparison.json`.

## 5. Results

Echo (τ<sub>clean</sub> = 0.7) against the action-only baseline on all 48,030
trials. The difference is in percentage points; CI is the 95% scene-bootstrap
interval.

| Backbone | Baseline | Echo | Difference | 95% CI | p (sign-flip) | p (cluster z) |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| PI05 | 20,145 (41.94%) | 20,381 (42.43%) | +0.49 | [+0.09, +0.89] | 0.015 | 0.014 |
| GR00T N1.5 | 18,203 (37.90%) | 18,927 (39.41%) | +1.51 | [+1.06, +1.97] | < 10<sup>-6</sup> | 1.1 × 10<sup>-10</sup> |

Echo beats the baseline in 1,451 scenes and loses in 1,339 for PI05. For GR00T
it wins in 1,563 scenes and loses in 1,322. Each model has one training seed, so
the p-values cover evaluation noise only. GR00T seed-to-seed variation is about
±2 percentage points.

## Tests and license

```bash
python -m unittest discover -s tests/echo -v
```

[Third-party provenance](THIRD_PARTY.md) lists the retained components and their
licenses.
