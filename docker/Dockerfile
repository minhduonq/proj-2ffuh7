# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:/opt/vllm-env/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib64:$LD_LIBRARY_PATH \
    PUPPETEER_CACHE_DIR=/opt/puppeteer-cache \
    NODE_MAJOR=20 \
    RENDER_POOL_SIZE=32

# System deps + Node.js 20 + Chromium runtime libs
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget git build-essential bash gnupg \
        # headless Chromium runtime
        fonts-liberation fonts-noto-color-emoji \
        libnss3 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 \
        libxdamage1 libxrandr2 libgbm1 libasound2 \
        libpangocairo-1.0-0 libgtk-3-0 \
        libegl1 libgl1-mesa-dri libgles2 \
        xvfb; \
    # Node.js 20 LTS via NodeSource
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -; \
    apt-get install -y --no-install-recommends nodejs; \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    speedtest-cli 

# Python deps
COPY docker/requirements.vllm.txt /tmp/requirements.vllm.txt
COPY docker/requirements.txt      /tmp/requirements.txt
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/vllm-env && \
    /opt/vllm-env/bin/pip install pip==26.2.1 && \
    /opt/vllm-env/bin/pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements.lock.txt

# Node deps 
# ---- 404 pgh1, leader style (19/10): only code is added here. /opt/ls/build_envs.sh (called by run.sh at container start,
# like the leader's own GLM env build) builds the SGLang coder env (lock file), the GDN kernel patches, the GLM vLLM judge
# env (the leader's script) and the judge scheduler patch, with retries; a failure keeps every server down (/status=replace).
ENV FLASHINFER_CACHE_DIR=/root/.cache/huggingface/.engine/flashinfer \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/huggingface/.engine/inductor \
    TRITON_CACHE_DIR=/root/.cache/huggingface/.engine/triton \
    SGLANG_CACHE_DIR=/root/.cache/huggingface/.engine/sglang
COPY docker/opt/ls/ /opt/ls/
COPY docker/opt/sgl/ /opt/sgl/
COPY docker/opt/kern/ /opt/kern/
COPY docker/opt/jf/ /opt/jf/
COPY docker/opt/prefetch/ /opt/prefetch/
COPY docker/opt/v3/ /opt/v3/
COPY docker/opt/raff/ /opt/raff/
RUN chmod +x /opt/ls/*.sh /opt/sgl/*.sh /opt/kern/*.sh /opt/jf/*.sh /opt/prefetch/*.sh /opt/v3/*.sh /opt/v3/seed/*.sh /opt/raff/*.sh
WORKDIR /workspace
COPY docker/package.json docker/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

# Add the pipeline service
COPY pipeline_service     /workspace
COPY configuration.yaml   /workspace/configuration.yaml
RUN chmod +x /workspace/run.sh

EXPOSE 10006 8001 8002
CMD ["/workspace/run.sh"]
# run-time settings of the tested package (LOOP top-4 round robin, burst judges, uuid judge client)
ENV BURST_GLM_URLS="http://localhost:8031/v1,http://localhost:8032/v1"
ENV BURST_PROXY="http://localhost:8001"
ENV JUDGE_LB_MIGRATE="1"
ENV SAVE_DRAFTS_DIR=""
ENV SPEC_FINAL="0"
ENV SAVE_SEMIS_DIR=""
ENV SAVE_SEMIS_MIN_ROUND="1"
ENV JUDGE_STOP_ISSUES="0"
ENV JUDGE_KEEPALIVE2="0"
ENV RENDER_DEADLINE_S="0"
ENV RENDER_DEADLINE_MIN_LIVE="20"
ENV ROUND_ROBIN_TOP4="1"
ENV ROUND_ROBIN_CUTOFF_S="2400"
ENV JUDGE_IMG_UUID="1"
# v2 RENDER_AFF: render sidecars pinned to floor(cpu quota) - RESERVE CPUs (refused below MIN -> plain node)
ENV RENDER_AFF_RESERVE="14"
ENV RENDER_AFF_MIN="8"
# v3 switches (pgh1v3_README.txt). Read at container start; override with docker run -e NAME=value, no rebuild.
# Coder chain: kern_shim.sh -> /opt/v3/coder_switch.sh -> /opt/sgl/sgl_shim.sh (seed patch + extra SGLang flags).
ENV KERN_REAL_SHIM="/opt/v3/coder_switch.sh"
# 1 = dflash_seed.patch applied at coder start + --sampling-backend pytorch (output-changing; speed cost unmeasured)
ENV SGLANG_DFLASH_SEEDED_SAMPLING="1"
# 1 = --enable-deterministic-inference (output-changing; slower, unmeasured)
ENV PGH_DETERMINISTIC_INFERENCE="0"
# 1 = abort the coder start if the seed patch is refused (default: fall back to the unpatched start + a !! line)
ENV PGH_SEED_STRICT="0"
# coder K (ensemble_size), adaptive_k.py: fixed = PGH_K40 on every pod | adaptive = max(PGH_K40, rule(cores)) <= 80
# | rule = the CPU-budget rule alone | off = the configured K 68 (= pgh1v2)
ENV PGH_K_MODE="fixed"
# K for a 40.8-CPU pod (fixed: K everywhere; adaptive: the floor). Bad value -> !! + the build default
ENV PGH_K40="74"
# legacy switch: anything but 1 (e.g. 0) = off, overriding PGH_K_MODE; 1 = PGH_K_MODE decides
ENV PGH_ADAPTIVE_K="1"
# JSON object overriding any of the PROVISIONAL rule coefficients (adaptive_k.DEFAULT_COEFFS keys: target_s, kappa,
# overhead_s, coder_s_per_draft, render_core_s_per_draft, k_min, k_max, k_fallback, ...); "" = defaults
ENV PGH_ADAPTIVE_K_COEFFS=""
# round-budget guard at batch boundaries (pgh_timeguard.py): live = cut K between batches when the round would
# overrun (output-changing when it fires) | shadow = log what live would do, change nothing | off = = pgh1v2
ENV PGH_TIMEGUARD="live"
# JSON object overriding the guard params (round_s, margin_s, batches_per_round, steps_pct, k_floor, idle_reset_s)
ENV PGH_TIMEGUARD_PARAMS=""
# 1 = A+ render fix in the js_checker serialize worker (output-changing: rejects non-texture map slots, bakes
# vertex-colour geometries); 0 = byte-identical to the unpatched worker. OFF until its duel test.
ENV PGH_APLUS="0"
