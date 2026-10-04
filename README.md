# AI 3D Worker — Image → Three.js Scene Pipeline

Worker nhận vào một **ảnh tham chiếu** (reference image) và sinh ra **code Three.js** (file `.js`) dựng lại scene 3D giống ảnh nhất có thể. Worker chạy như một HTTP service: orchestrator bên ngoài gửi batch prompt vào, worker chạy pipeline multi-agent (LLM sinh code → validate → render → chấm điểm), rồi trả về file `.js` tốt nhất cho từng prompt.

---

## Mục lục

1. [Ý tưởng & cách tiếp cận](#1-ý-tưởng--cách-tiếp-cận)
2. [Kiến trúc tổng quan](#2-kiến-trúc-tổng-quan)
3. [Các thành phần cốt lõi](#3-các-thành-phần-cốt-lõi)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Luồng hoạt động](#5-luồng-hoạt-động)
6. [API endpoints](#6-api-endpoints)
7. [Cấu hình — configuration.yaml](#7-cấu-hình--configurationyaml)
8. [Hướng dẫn chạy](#8-hướng-dẫn-chạy)
9. [Test & xem kết quả](#9-test--xem-kết-quả)
10. [Debug & troubleshooting](#10-debug--troubleshooting)

---

## 1. Ý tưởng & cách tiếp cận

**Bài toán:** từ một ảnh 2D, dựng lại scene 3D. Thay vì sinh trực tiếp mesh/point cloud, repo chọn hướng **sinh code Three.js** — output là chương trình, nên có thể parse, chạy thử, render và chấm điểm hoàn toàn tự động.

**Vấn đề:** LLM sinh code không đáng tin ở lần chạy đầu — code có thể lỗi syntax, dùng API không tồn tại, hoặc chạy được nhưng dựng scene sai hoàn toàn so với ảnh.

**Lời giải — generate → validate → render → compare:**

1. **Sinh nhiều candidate (ensemble)**: coder LLM sinh **N candidate song song** (hiện tại N=60) với seed/temperature khác nhau, thay vì phụ thuộc vào một lần sinh duy nhất.
2. **Validate bằng tool deterministic, không bằng LLM**: mỗi candidate đi qua một validator (Babel parse, static analysis theo allowlist Three.js, chạy trong sandbox có timeout/heap cap). Code lỗi bị loại hoặc đưa lại cho coder tự sửa (`code_repair`).
3. **Đánh giá dựa trên ảnh render thật**: candidate hợp lệ được render thành ảnh PNG bằng Chromium headless. Mọi đánh giá chất lượng đều dựa trên ảnh render — không dựa vào việc LLM tự nhận xét code của chính nó.
4. **So sánh theo cặp (pairwise) thay vì chấm điểm tuyệt đối**: một vision LLM (judge) so từng **cặp** ảnh render với ảnh gốc theo bracket loại trực tiếp (single-elimination) để chọn candidate tốt nhất — trả lời "A hay B giống ảnh hơn?" đáng tin cậy hơn so với yêu cầu model chấm một điểm số tuyệt đối.

**Nguyên tắc thiết kế xuyên suốt:**

- **Tách agent LLM khỏi module deterministic.** Agent (planner/coder/critic/judge) là non-deterministic, được kiểm soát bằng seed + sampling params. Module (js_checker, renderer) là deterministic — cùng input luôn ra cùng output. Nhờ đó kết quả reproducible theo seed và debug được từng stage độc lập.
- **Config-driven topology.** Toàn bộ hình dạng pipeline (bật/tắt planner, cỡ ensemble, có refinement hay không, model nào chạy ở đâu) nằm trong `configuration.yaml`; pydantic validator chặn các tổ hợp config vô nghĩa ngay lúc khởi động.
- **Fault tolerance ở mọi tầng**: candidate fail → walkover; task fail → retry 1 lần; sidecar render chết → watchdog restart; GPU lỗi → đánh dấu pod cần thay. Mọi failure đều gắn `failure_stage` để truy ngược.

## 2. Kiến trúc tổng quan

```
                       ┌────────────────────────────────────────────────┐
                       │              FastAPI  (:10006)                 │
 Orchestrator ───────▶ │  /generate  /status  /results  /debug/*        │
 (bên ngoài)           └───────────────────┬────────────────────────────┘
                                           │ batch tasks
                                           ▼
                       ┌────────────────────────────────────────────────┐
                       │           GenerationPipeline / Pipeline        │
                       │                                                │
                       │  prepare ─▶ coder ─▶ js_checker ─▶ renderer    │
                       │  (planner)   │                        │        │
                       │              │      ┌─────────────────┘        │
                       │              ▼      ▼                          │
                       │           critic / judge  ──▶ best snapshot    │
                       └──────┬──────────────┬──────────────┬───────────┘
                              │              │              │
                    ┌─────────▼───┐  ┌───────▼──────┐  ┌────▼─────────────┐
                    │ vLLM coder  │  │ vLLM judge/  │  │ Node sidecars    │
                    │ Qwen :8001  │  │ critic GLM   │  │ Puppeteer render │
                    │ (GPU 0,1,2) │  │ :8002 (3)    │  │ :8003+ (headless)│
                    └─────────────┘  └──────────────┘  └──────────────────┘
```

Ba lớp, tách biệt rõ ràng:

| Lớp | Vai trò | Thành phần chính |
|---|---|---|
| **API layer** | Giao tiếp với orchestrator: nhận batch, báo trạng thái qua state machine (`warming_up → ready → generating → complete`), trả kết quả | `serve.py`, `WorkerState` |
| **Pipeline layer** | Điều phối vòng đời từng task: gọi agent LLM và module deterministic theo đúng thứ tự, quản lý concurrency, retry, deadline | `generation_pipeline.py`, `orchestrator.py`, `stages.py` |
| **Serving layer** | Chạy model: 2 vLLM server local (coder Qwen + judge/critic GLM, mỗi model một venv riêng do xung đột version `transformers`), hoặc OpenRouter khi chạy cloud-only | `llm/spawn.py`, `run.sh` |

## 3. Các thành phần cốt lõi

### Agent LLM (`modules/scene_planner`, `scene_coder`, `critic`, `judge`)

Mỗi agent cùng một layout `agent.py + prompts.py + schema.py`, xây trên `base_agent.py`, nhận sampling params từ `ActorConfig` trong YAML:

| Agent | Model | Nhiệm vụ | Ghi chú |
|---|---|---|---|
| **Planner** | vision | Nhìn ảnh → viết scene description có cấu trúc (OSD) cho coder | Optional, **đang tắt** — coder nhìn ảnh trực tiếp (`multimodal: true`) |
| **Coder** | Qwen (code) | Sinh code Three.js từ ảnh/OSD; có 3 chế độ: sinh mới, `code_repair` (sửa lỗi checker), `code_critic_repair` (sửa theo report của critic) | Chạy ensemble N candidate song song |
| **Judge** | GLM (vision) | Mỗi match nhìn ảnh gốc + 2 ảnh render → chọn A/B; đấu bracket để tìm candidate tốt nhất | Chỉ chạy khi `ensemble_size > 1` |
| **Critic** | GLM (vision) | Chấm điểm ảnh render vs ảnh gốc (0→1) + liệt kê lỗi cho vòng refinement | Chỉ chạy khi `refinement_enabled: true` — **đang tắt** |

### Module deterministic (`modules/js_checker`, `renderer`, `monitoring`, `metrics`)

| Module | Nhiệm vụ | Cách hoạt động |
|---|---|---|
| **js_checker** | Validate code sinh ra trước khi render | Spawn Node subprocess: Babel parse → static analysis theo allowlist Three.js (chặn API cấm) → chạy thật trong worker thread có timeout 5s + heap cap → đo metrics (vertices, draw calls, depth) |
| **renderer** | Render scene code thành ảnh PNG | Pool sidecar HTTP (Node/Puppeteer, port 8003+), render scene thành PNG grid 2×2 nhiều góc nhìn bằng Chromium headless; có watchdog tự restart sidecar chết |
| **monitoring** | Health check các vLLM server | Probe `/health` + model list của từng vLLM server bằng subprocess, cập nhật `WorkerState.llm_status` để state machine biết khi nào `ready` |
| **metrics** | GPU preflight | Benchmark GPU lúc khởi động; fail → tạo marker `/tmp/pod_replace` báo orchestrator thay pod |

### Đối tượng trung tâm: `PipelineTask` (`pipeline/task.py`)

Một prompt = một `PipelineTask` — object mutable được truyền xuyên suốt mọi stage, gom toàn bộ dữ liệu của task: ảnh gốc (`image_bytes`), OSD, `js_code`, ảnh render, điểm số, timing từng stage, và thông tin lỗi (`failure_stage`, `failure_reason`). Debug một task = xem object này qua `/debug/tasks/{stem}`.

## 4. Cấu trúc thư mục

```
.
├── configuration.yaml          # TOÀN BỘ config runtime (model, GPU, sampling, port…)
├── docker/
│   ├── Dockerfile              # Image PyTorch + Node 20 + Chromium libs
│   ├── requirements.txt        # Python deps của service
│   ├── requirements.vllm.txt   # Deps của venv vLLM mặc định
│   └── package.json            # Node deps: three, puppeteer, @babel/parser
├── pipeline_service/           # ⚠️ Mọi Python import tính từ gốc thư mục này
│   ├── serve.py                # FastAPI app — entry point của service
│   ├── run.sh                  # Entry point khi deploy: preflight → API → vLLM
│   ├── config/settings.py      # Load configuration.yaml → pydantic singleton
│   ├── pipeline/
│   │   ├── generation_pipeline.py  # Driver cấp app: LLM client, batch, retry
│   │   ├── orchestrator.py         # Vòng đời 1 task: iterate coder→check→render→critic
│   │   ├── stages.py               # Từng stage + multigen ensemble + judge bracket
│   │   ├── factory.py              # Wire agent ↔ client theo config
│   │   ├── task.py                 # PipelineTask — object dữ liệu xuyên suốt pipeline
│   │   └── state.py                # WorkerState + state machine
│   ├── modules/
│   │   ├── scene_planner/  scene_coder/  critic/  judge/   # Agent LLM (agent+prompts+schema)
│   │   ├── js_checker/     # Validator chạy Node (Babel, allowlist, sandbox)
│   │   ├── renderer/       # Pool sidecar Puppeteer (render PNG)
│   │   ├── monitoring/     # Probe health/model của vLLM
│   │   └── metrics/        # GPU preflight benchmark
│   ├── llm/spawn.py        # Đọc config → chạy `vllm serve` cho từng client local
│   └── scripts/setup_glm_vllm_env.sh   # Build venv vLLM riêng cho GLM
└── tests/
    ├── test_pipeline.py    # Smoke test end-to-end qua HTTP
    ├── prompts/test_prompts.txt        # Danh sách "stem  image_url"
    ├── gen_results.py      # Tạo manifest results.json cho viewer
    └── viewer.html         # Xem file .js đã sinh ngay trên trình duyệt
```

## 5. Luồng hoạt động

### 5.1. Khởi động service (state machine)

Trạng thái của worker (`WorkerStatus`): `warming_up → ready → generating → complete` (+ `replace` khi hỏng phần cứng).

1. **`run.sh`** chạy GPU preflight benchmark (`python -m modules.metrics.preflight`). Nếu fail → tạo marker `/tmp/pod_replace`, service khởi động ở trạng thái `replace` (báo orchestrator thay pod), không chạy vLLM.
2. FastAPI (`serve.py`) khởi động ở trạng thái `warming_up`. `GenerationPipeline.startup()` tạo các OpenAI client, khởi động js_checker + pool renderer sidecar, wire pipeline.
3. `run.sh` build (nếu thiếu) 2 venv vLLM rồi chạy `python -m llm.spawn` — spawn một process `vllm serve` cho **mỗi** client local trong `llm_clients` (coder trên :8001, judge/critic trên :8002), gán GPU theo `gpu_ids`.
4. Vòng lặp `_watch_models` (modules/monitoring) probe `/health` + model list của từng vLLM server. Khi **tất cả** RUNNING:
   - Nếu `warmup: true` — chạy 1 task warmup (ảnh `warmup.png` nội bộ) xuyên hết pipeline để warm up model và browser.
   - Chuyển trạng thái sang **`ready`**.

> Khi chạy **cloud-only** (chỉ dùng OpenRouter, không có client local nào): không có health probe, service `ready` ngay.

### 5.2. Nhận batch

```
POST /generate  { "prompts": [{"stem": "...", "image_url": "..."}, ...], "seed": 42 }
```

- Chỉ nhận khi trạng thái là `ready` hoặc `complete`. Đang `generating` mà nhận đúng bộ stems cũ → trả accepted luôn (idempotent); bộ stems khác → 409.
- Batch mới hủy task của batch cũ, `WorkerState.reset_for_batch()`. Seed của batch được gán cho mọi agent để kết quả reproducible.
- Mỗi prompt thành một `PipelineTask`, chạy song song (`asyncio.gather`) trong giới hạn `batch_time_budget` (mặc định 4h, trừ 120s buffer). Task fail được **retry đúng 1 lần** bằng một task mới hoàn toàn.
- Xong batch → trạng thái `complete`, dọn session/memory.

### 5.3. Vòng đời một task

Mỗi task có deadline riêng `event_bus.task_deadline_s` (3600s). Concurrency từng stage bị giới hạn bởi semaphore theo `actors.*.workers`.

**Bước 1 — prepare**: tải ảnh về (`image_bytes`). Nếu bật planner → gọi planner sinh OSD (scene description dạng JSON). Config hiện tại tắt planner → coder dùng ảnh trực tiếp.

**Bước 2 — iteration 0, multigen** (khi `ensemble_size > 1`, hiện tại là 60):

```
           ┌─ coder k=0 ─ dedup ─ js_check ─ render ─┐
 ảnh gốc ──┼─ coder k=1 ─ dedup ─ js_check ─ render ─┼──▶ judge bracket ──▶ winner
           ├─ ...        (code trùng → walkover)     │    (single-elimination,
           └─ coder k=59 ─ ...                       ┘     so từng cặp ảnh render)
```

- 60 coder chạy song song, mỗi candidate có seed riêng (`seed + k`) và temperature riêng (`ensemble_temperature`).
- Validation (js_check → render) của mỗi candidate chạy overlap với vòng quét dedup — candidate không phải chờ mọi coder trước nó xong mới được validate.
- Candidate nào có code trùng byte-identical với candidate trước → được gộp lại (dedup), không tốn lượt render/judge.
- Candidate fail check hoặc fail render → bị loại (đối thủ được walkover).
- **Judge bracket**: đấu loại trực tiếp theo vị trí cố định — mỗi match, judge nhìn ảnh gốc + 2 ảnh render (kèm multiview và DINO embedding nếu bật) rồi chọn A/B. Match bắt đầu ngay khi 2 nhánh con xong → judge chạy song song với render, không phải chờ cả vòng.
- Tất cả candidate đều fail → task fail với stage `multigen`.
- Winner được ghi vào task; session của các candidate thua bị giải phóng.

**Bước 3 — refinement loop** *(hiện đang TẮT — `refinement_enabled: false`)*: khi bật, sau mỗi iteration critic chấm điểm ảnh render (0→1) và liệt kê lỗi; nếu điểm < `score_threshold` và còn lượt (`max_iter`), coder chạy chế độ `code_critic_repair` sửa theo report, rồi lại check → render → critic. Snapshot có điểm cao nhất giữa các iteration được giữ làm kết quả cuối. Khi tắt, kết quả của iteration 0 là kết quả cuối, bỏ qua critic hoàn toàn.

**Vòng sửa lỗi JS (bên trong mọi iteration)**: nếu js_checker báo lỗi, coder chạy chế độ `code_repair` với danh sách lỗi và lặp đến khi code hợp lệ (trong giới hạn deadline của task).

### 5.4. Trả kết quả

```
GET /results   → file ZIP: mỗi task thành công 1 file {stem}.js
                 + _failed.json liệt kê task fail và lý do
```

Chỉ trả khi trạng thái `complete` (trạng thái khác → 409).

## 6. API endpoints

| Method | Path | Mô tả |
|---|---|---|
| GET | `/health` | Liveness — luôn 200 khi process còn sống |
| GET | `/status` | `{status, progress, total}` — orchestrator poll endpoint này |
| POST | `/generate` | Nhận batch `{prompts: [{stem, image_url}], seed}` → `{accepted: n}` |
| GET | `/results` | ZIP kết quả (chỉ khi `complete`) |
| GET | `/debug/tasks` | Tổng quan mọi task trong batch: điểm, timing, metrics |
| GET | `/debug/tasks/{stem}` | Chi tiết 1 task: js_code, ảnh render base64, OSD, lỗi… |
| GET | `/debug/logs` | Tải file log `logs/pipeline.log` |

## 7. Cấu hình — configuration.yaml

File duy nhất điều khiển toàn bộ runtime, được load **một lần lúc import** thành singleton (`config/settings.py`) — sửa config xong phải restart service.

```yaml
api:            # host/port FastAPI (mặc định :10006)
pipeline:
  batch_time_budget: 14400   # ngân sách thời gian cả batch (giây)
  use_planner: false         # tắt planner → coder nhìn ảnh trực tiếp
  render_from_object: true   # render từ object đã serialize (checker mode with_object)
  refinement_enabled: false  # tắt vòng critic/refinement

llm_clients:    # các LLM endpoint, đặt tên tự do
  coder-instance:            # base_url localhost → hiểu là vLLM local, sẽ được spawn
    base_url: "http://localhost:8001/v1"
    vllm: { model, revision, gpu_ids, tensor_parallel_size, vllm_bin, ... }
  judge-critic-instance:     # GLM-4.6V — venv vLLM riêng vì cần transformers mới hơn
    base_url: "http://localhost:8002/v1"
  openrouter:                # phương án cloud, bật bằng enabled: true + OPENROUTER_API_KEY
    enabled: false

actors:         # mỗi actor: dùng client nào, model nào, sampling params, số worker song song
  coder:   { client: coder-instance, ensemble_size: 60, multimodal: true, ... }  # extra_clients: [...] để load-balance nhiều vLLM instance
  critic:  { client: judge-critic-instance, ... }
  judge:   { client: judge-critic-instance, ... }

event_bus:
  max_iter: 2                # số iteration refinement tối đa
  task_deadline_s: 3600      # deadline mỗi task
  score_threshold: 0.55      # điểm critic đạt ngưỡng thì dừng sớm

renderer:       # pool sidecar Puppeteer: số sidecar, port, timeout, kích thước ảnh
js_checker:     # timeout chạy sandbox, đường dẫn node
```

**Pydantic validator sẽ chặn ngay lúc khởi động nếu config sai**, ví dụ: `use_planner: false` mà `coder.multimodal` không bật; `ensemble_size > 1` mà thiếu judge; actor trỏ tới client không tồn tại hoặc đang tắt.

## 8. Hướng dẫn chạy

### 8.1. Yêu cầu

| Chế độ | Yêu cầu |
|---|---|
| **Full local (production)** | 4× GPU (config hiện tại: coder dùng GPU 0,1,2 · GLM dùng GPU 3), CUDA 12.8, Docker |
| **Cloud-only (dev/thử nghiệm)** | Không cần GPU — chỉ cần OpenRouter API key |
| Chung | Python 3.10+, Node.js 20, Chromium libs (Puppeteer tự tải Chromium) |

> **Windows:** chỉ **cloud-only ([Cách 2](#83-cách-2--dev-local-không-docker-cloud-only-không-cần-gpu))** chạy được native trên Windows — và nó chạy đầy đủ (FastAPI + js_checker Node + renderer Puppeteer + OpenRouter). Phần serving local (`run.sh`, vLLM, CUDA) là bash/Linux-only, muốn chạy thì dùng **WSL2** hoặc **Docker**. Mọi lệnh Windows trong tài liệu này viết cho **PowerShell** (không phải cmd.exe); xem [mục 8.5](#85-ghi-chú-riêng-cho-windows) cho các khác biệt cần biết trước khi chạy.

### 8.2. Cách 1 — Docker (khuyến nghị, giống môi trường production nhất)

**Bước 1 — build image** (từ gốc repo):

```bash
docker build -f docker/Dockerfile -t ai3d-worker .
```

Lệnh này giống nhau trên mọi OS (Windows cần Docker Desktop với backend WSL2; muốn dùng `--gpus all` thì phải có NVIDIA driver + WSL2 GPU support).

**Bước 2 — chạy container** (cần GPU và mở đủ port):

```bash
# Linux / macOS
docker run --gpus all \
  -p 10006:10006 -p 8001:8001 -p 8002:8002 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  ai3d-worker
```

```powershell
# Windows PowerShell — nối dòng bằng backtick (`), không phải backslash (\)
docker run --gpus all `
  -p 10006:10006 -p 8001:8001 -p 8002:8002 `
  -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" `
  -e OPENROUTER_API_KEY="$env:OPENROUTER_API_KEY" `
  ai3d-worker
```

Mount `~/.cache/huggingface` không bắt buộc nhưng nên có: model nặng vài chục GB, có cache thì các lần khởi động sau không phải tải lại.

Container tự chạy `run.sh` theo thứ tự: GPU preflight → FastAPI → build venv vLLM (lần đầu khá lâu: tải wheel vLLM + tải model từ HuggingFace) → spawn 2 vLLM server.

**Bước 3 — đợi service sẵn sàng:**

```bash
# Linux / macOS
curl localhost:10006/status
# {"status":"warming_up"...}  → đợi (lần đầu có thể rất lâu)
# {"status":"ready"...}       → sẵn sàng nhận batch
```

```powershell
# Windows PowerShell — `curl` là alias của Invoke-WebRequest, cú pháp khác hẳn.
# Dùng curl.exe (bản thật) hoặc Invoke-RestMethod:
curl.exe localhost:10006/status
Invoke-RestMethod http://localhost:10006/status
```

### 8.3. Cách 2 — Dev local không Docker (cloud-only, không cần GPU)

Phù hợp khi muốn sửa code pipeline mà không có GPU. Model chạy qua OpenRouter thay vì vLLM local. Đây cũng là **cách duy nhất chạy native trên Windows**.

**Bước 1 — cài Python deps** (từ gốc repo):

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
pip install -r docker/requirements.txt
pip install python-dotenv          # để serve.py đọc file .env
```

```powershell
# Windows PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1       # nếu bị chặn: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r docker/requirements.txt
pip install python-dotenv
```

`docker/requirements.txt` **không có** `torch`/`transformers`/`pillow`. Thiếu chúng thì judge vẫn chạy nhưng tự tắt phần DINO embedding (log `[DINO] dependencies unavailable ...; S2BV disabled`). Muốn bật đầy đủ: `pip install torch transformers pillow` — bản CPU là đủ cho dev, lần dùng đầu sẽ tải model DINO (~90 MB) từ HuggingFace.

**Bước 2 — cài Node deps.** js_checker và renderer spawn process `node`, nên `node_modules` phải resolve được từ thư mục chạy service (`pipeline_service/`):

```bash
# Linux / macOS
cd pipeline_service
cp ../docker/package.json ../docker/package-lock.json .
npm ci
```

```powershell
# Windows PowerShell — không có `cp` với 2 nguồn; dùng Copy-Item với danh sách ngăn bởi dấu phẩy
cd pipeline_service
Copy-Item ..\docker\package.json, ..\docker\package-lock.json .
npm ci                             # nếu npm.ps1 bị ExecutionPolicy chặn: npm.cmd ci
```

Lần chạy `npm ci` đầu tiên, Puppeteer tải Chromium về `%USERPROFILE%\.cache\puppeteer` (Windows) hoặc `~/.cache/puppeteer` — vài trăm MB, cần mạng.

**Bước 3 — sửa `configuration.yaml`** để chuyển sang OpenRouter. **Phải đổi cả 4 actor**, không chỉ coder:

```yaml
warmup: false                      # dev: bỏ warmup cho nhanh

llm_clients:
  openrouter:            { enabled: true }        # bật cloud
  coder-instance:        { enabled: false }       # tắt 2 client local
  judge-critic-instance: { enabled: false }

actors:
  planner: { client: "openrouter", model: "google/gemini-3.5-flash" }
  coder:   { client: "openrouter", model: "google/gemini-3.5-flash", ensemble_size: 2 }
  critic:  { client: "openrouter", model: "z-ai/glm-4.6v" }
  judge:   { client: "openrouter", model: "google/gemini-3.5-flash" }

renderer:
  sidecar_count: 2                 # dev: 8 sidecar × pool_size 2 = 16 Chromium, quá nặng cho laptop
```

**Chọn model cho judge — không tuỳ ý được.** `modules/judge/multi_stage.py` gọi `response_format={"type": "json_schema", ...}`, nên model của judge phải là **vision + hỗ trợ structured outputs**. Kiểm tra trước khi chọn:

```powershell
# model nào có cả "image" input lẫn "structured_outputs"?
Invoke-RestMethod https://openrouter.ai/api/v1/models |
  Select-Object -ExpandProperty data |
  Where-Object { $_.architecture.input_modalities -contains "image" -and
                 $_.supported_parameters -contains "structured_outputs" } |
  Select-Object id, @{n="in";e={$_.pricing.prompt}} | Format-Table
```

`z-ai/glm-4.6v` **trượt bài kiểm tra này** (`structured_outputs: false`) dù nó chính là bản cloud của model GLM chạy local — OpenRouter route sang provider Novita là trả `400 INVALID_REQUEST_BODY: Model does not support 'json_schema' response format`. Và **không chữa được bằng `actors.judge.providers`**: judge gọi OpenAI client trực tiếp, không đẩy `provider` routing vào `extra_body` như coder/critic. Đổi model là cách duy nhất.

**Chất lượng cloud-only sẽ thấp hơn bản full local — đừng so trực tiếp.** Coder mặc định của production là `Tooony133/Qwen-3.6-27B-ChonkyCat`, một **fine-tune riêng cho bài toán này** (repo HF riêng, revision ghim cứng). Không model general-purpose nào trên OpenRouter khớp được prior đó. Ba yếu tố quyết định khoảng cách, theo thứ tự ảnh hưởng:

1. **Model coder.** Chênh lệch giữa các model general-purpose cũng đã rất lớn. A/B cùng prompt, cùng ảnh, 3 seed, đo bằng chính js_checker:

   | Coder model | vertices | draw calls | Kết quả nhìn bằng mắt |
   |---|---|---|---|
   | `google/gemini-3.5-flash` | 7.7k – 9.7k | 8 – 13 | Đúng tỉ lệ, có bo góc, có lỗ treo, có kẹp dây thép — cả 3/3 seed |
   | `qwen/qwen3-vl-235b-a22b-instruct` | 372 – 454 | 5 – 6 | Tấm phẳng, sai tỉ lệ, không có kẹp — cả 3/3 seed |

   Cả hai đều **pass** js_checker, nên checker không lọc ra được sự khác biệt này — chỉ judge (hoặc mắt người) mới thấy.

2. **`ensemble_size`.** Cơ chế chất lượng cốt lõi của repo là best-of-N + judge bracket. Production chạy N=60; hạ xuống 2–4 cho rẻ thì mất gần hết phần "chọn cái tốt nhất", không phải lỗi pipeline.

3. **Sampling params rơi rụng.** `top_k`, `min_p`, `repetition_penalty` và `enable_thinking` là tham số vLLM; qua OpenRouter, provider nào không hỗ trợ sẽ **âm thầm bỏ qua** (xem `supported_parameters` ở endpoint `/models/{id}/endpoints`). Phần lớn provider cũng phục vụ bản lượng tử hoá fp8 thay vì bf16.

> Nói cách khác: chế độ cloud-only dùng để **phát triển và debug pipeline**, không dùng để tái hiện chất lượng production.

Ba điểm hay làm sai ở bước này:

- **`critic` vẫn trỏ `judge-critic-instance`** → crash ngay lúc start: `ValidationError: actors.critic.client=... points to a disabled client`. Validator bắt buộc `critic.client` phải là client đang bật **kể cả khi `refinement_enabled: false`** (critic nằm trong danh sách actor bắt buộc, xem `_LLM_ACTORS_BASE` trong `config/settings.py`). Đây là lỗi hay gặp nhất khi làm theo hướng dẫn cũ vì mục này chỉ nhắc đổi coder.
- **`judge` vẫn trỏ `judge-critic-instance`** → cũng crash y hệt, bắt buộc mỗi khi `actors.coder.ensemble_size > 1`.
- **Giữ nguyên tên model của bản local** → service start được nhưng mọi call LLM trả 404. `Tooony133/Qwen-3.6-27B-ChonkyCat` và `zai-org/GLM-4.6V-Flash` là **HF repo id cho vLLM**, OpenRouter không có. Phải đổi sang model id của OpenRouter, và **coder/critic/judge đều cần model VISION** vì `multimodal: true`.

**Bước 4 — đặt API key.** `serve.py` gọi `load_dotenv()`, nên đơn giản nhất là tạo file `.env` ở **gốc repo** (đã nằm trong `.gitignore`):

```
OPENROUTER_API_KEY=sk-or-...
```

Hoặc set biến môi trường trực tiếp trong shell:

```bash
# Linux / macOS
export OPENROUTER_API_KEY=sk-or-...
```

```powershell
# Windows PowerShell — không có `export`; biến chỉ sống trong session hiện tại
$env:OPENROUTER_API_KEY = "sk-or-..."
```

**Bước 5 — chạy service.** Bắt buộc đứng trong `pipeline_service/` vì mọi import Python là absolute từ đây:

```bash
# Linux / macOS
cd pipeline_service        # nếu chưa ở trong đó
python serve.py            # serve trên :10006
```

```powershell
# Windows PowerShell
cd pipeline_service
$env:PYTHONUTF8 = "1"      # khuyến nghị, bắt buộc nếu redirect log ra file — xem mục 8.5
python serve.py            # serve trên :10006
```

Kiểm tra ở terminal thứ 2: `curl.exe localhost:10006/status` → `{"status":"ready",...}`.

Không có client vLLM local → không có health probe → service `ready` gần như ngay (sau warmup nếu bật).

> **Mẹo cho vòng lặp dev nhanh:** đặt `actors.coder.ensemble_size: 1` (bỏ multigen + judge bracket) và `warmup: false` — mỗi lần thử sẽ nhanh hơn rất nhiều.
>
> ⚠️ Đây không chỉ là chuyện tốc độ mà còn là **tiền OpenRouter**: config mặc định để `ensemble_size: 60` + `warmup: true`, nghĩa là service gọi **60 lần coder cho riêng ảnh warmup** trước khi chuyển sang `ready`, rồi 60 lần nữa cho mỗi prompt trong batch.

### 8.4. Cách 3 — Full local không Docker

Như cách 2 nhưng giữ nguyên 2 client local và tự lo phần model serving:

```bash
bash pipeline_service/run.sh
```

Script tự build 2 venv vLLM vào `/opt/vllm-coder-env` và `/opt/vllm-glm-env` rồi spawn model. Máy cần đúng CUDA 12.8 và đủ VRAM cho cả 2 model.

> **Không chạy được native trên Windows.** `run.sh` là bash script với heredoc/`sed`, đường dẫn `/opt/...`, `/tmp/pod_replace`, và vLLM chỉ build cho Linux+CUDA. Trên Windows hãy dùng **WSL2** (`wsl bash pipeline_service/run.sh`) hoặc [Cách 1 — Docker](#82-cách-1--docker-khuyến-nghị-giống-môi-trường-production-nhất).

### 8.5. Ghi chú riêng cho Windows

Những khác biệt đã được kiểm chứng khi chạy Cách 2 trên Windows 11 + PowerShell + Python 3.13 + Node 22:

| Vấn đề | Chi tiết |
|---|---|
| **`PYTHONUTF8=1`** | Log của pipeline có ký tự `→`, `█`, `─`. Khi stdout không phải console (redirect ra file, pipe qua `tee`, chạy trong CI) Python dùng code page `cp1252` → `UnicodeEncodeError: 'charmap' codec can't encode`. Triệu chứng: log server in ra khối `Record was: {...}` thay vì dòng log, và `tests/test_pipeline.py` **crash ngay ở thanh progress bar**. Set `$env:PYTHONUTF8 = "1"` trước khi chạy cả server lẫn test. |
| **`curl` ≠ `curl`** | Trong PowerShell, `curl` là alias của `Invoke-WebRequest` (tham số hoàn toàn khác). Dùng `curl.exe ...` hoặc `Invoke-RestMethod http://localhost:10006/status`. |
| **`export` / `&&` / `\`** | PowerShell dùng `$env:VAR = "..."` thay `export`; `;` hoặc `if ($?) {...}` thay `&&` (PowerShell 5.1 không có `&&`); backtick `` ` `` thay `\` để nối dòng. |
| **`cp` nhiều nguồn, `mkdir -p`** | `Copy-Item a, b dest` (dấu phẩy), `New-Item -ItemType Directory -Force dest`. |
| **Process mồ côi khi tắt server** | Windows không kill theo process group: sau `Ctrl+C` (hoặc khi process bị kill cứng), **các `node.exe` sidecar + hàng chục `chrome.exe` headless vẫn sống** và giữ port 10006/8003+. Lần chạy sau sẽ báo `[Errno 10048] only one usage of each socket address`. Xem lệnh dọn ở [mục 10](#10-debug--troubleshooting). |
| **Firewall** | Lần đầu chạy, Windows Defender hỏi cho phép `node.exe` và `python.exe` mở port. Chỉ cần cho phép **Private network** — sidecar chỉ bind `127.0.0.1`. |
| **`nvidia-smi` / GPU preflight** | Không dùng ở chế độ cloud-only: `benchmark`/preflight chỉ chạy qua `run.sh`, còn `serve.py` chỉ kiểm tra marker `/tmp/pod_replace` (trên Windows đường dẫn này gần như chắc chắn không tồn tại → bỏ qua, không lỗi). |

## 9. Test & xem kết quả

Không có unit test — kiểm thử bằng **smoke test end-to-end** qua HTTP, với điều kiện server đang chạy (mục 8).

```bash
# Linux / macOS — từ gốc repo. Mặc định đọc tests/prompts/test_prompts.txt (mỗi dòng: "stem  image_url")
python tests/test_pipeline.py --limit 2 --name myrun

# Server ở máy khác
python tests/test_pipeline.py --host 192.168.1.50 --port 10006 --seed 42
```

```powershell
# Windows PowerShell — từ gốc repo, mở terminal thứ 2 (terminal 1 đang chạy serve.py)
$env:PYTHONUTF8 = "1"        # thanh progress bar dùng ký tự █ → thiếu dòng này sẽ UnicodeEncodeError
python tests\test_pipeline.py --limit 2 --name myrun

# Server ở máy khác
python tests\test_pipeline.py --host 192.168.1.50 --port 10006 --seed 42
```

Script sẽ: đợi `/health` → đợi hết `warming_up` → POST `/generate` → poll `/status` (có progress bar) → tải kết quả từng task về `tests/prompts/myrun/`:

```
myrun/
├── {stem}.js          # code Three.js sinh ra
├── {stem}.png         # ảnh render grid 2×2
├── {stem}_input.png   # ảnh gốc
└── results.json       # metadata: điểm, timing, lỗi từng task
```

**Xem scene trên trình duyệt:**

```bash
# Linux / macOS
mkdir -p tests/results
cp tests/prompts/myrun/*.js tests/results/   # 1. copy các file .js vào tests/results/
python tests/gen_results.py                  # 2. tạo manifest tests/results.json
cd tests && python -m http.server            # 3. mở http://localhost:8000/viewer.html
```

```powershell
# Windows PowerShell — thư mục tests/results/ chưa tồn tại trong repo, phải tạo trước
New-Item -ItemType Directory -Force tests\results | Out-Null
Copy-Item tests\prompts\myrun\*.js tests\results\
python tests\gen_results.py
cd tests; python -m http.server              # mở http://localhost:8000/viewer.html
```

## 10. Debug & troubleshooting

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `/status` kẹt `warming_up` mãi | vLLM chưa lên (đang tải model / OOM / crash) | Xem log process vLLM; `curl localhost:8001/health` và `:8002/health` |
| Service khởi động ở `replace` | GPU preflight fail, có file `/tmp/pod_replace` | Kiểm tra GPU (`nvidia-smi`), xóa marker rồi restart |
| Crash ngay lúc start, lỗi pydantic | `configuration.yaml` vi phạm validator | Đọc message lỗi — nó nói rõ thiếu gì (xem mục 7) |
| `ModuleNotFoundError: config` | Chạy Python từ sai thư mục | Luôn `cd pipeline_service` trước khi chạy |
| Task fail ở stage `renderer` | Sidecar Puppeteer chết / thiếu Chromium libs | Xem log renderer; chạy ngoài Docker phải cài đủ system libs (tham khảo Dockerfile) |
| Task fail ở stage `multigen` "all candidates failed" | Model coder sinh code hỏng toàn bộ / checker quá chặt | Xem `/debug/tasks/{stem}` → `js_errors` của từng candidate trong log |
| `judge views \| white=0 gray=0 \| 180.1s` rồi `S1: missing white view 'front_left', skipping angle` → duel nào cũng `S1 -> draw` | **Không đủ slot render.** `180.1s` = đúng 2 × `renderer.request_timeout_s` (90s): cả lượt white lẫn gray đều timeout. Số slot = `sidecar_count × pool_size`; mỗi duel cần 9 ảnh view cho mỗi bên, `ensemble_size: 60` sinh 59 duel → hàng nghìn lượt render dồn vào vài slot | Cân lại 2 con số: hạ `actors.coder.ensemble_size` (dev: 2–8) **hoặc** nâng `renderer.sidecar_count`. Tham chiếu: 2 candidate + `sidecar_count: 8` → judge views xong trong ~15s. Bracket vẫn "chạy" khi thiếu view nhưng judge mất stage so góc nghiêng → kết quả gần như ngẫu nhiên |
| `s2cl_decompose ... 400 INVALID_REQUEST_BODY: Model does not support 'json_schema' response format` | Model của judge không có `structured_outputs` trên OpenRouter (điển hình: `z-ai/glm-4.6v` khi route sang Novita) | Đổi `actors.judge.model` sang model vision có `structured_outputs` — xem lệnh kiểm tra ở [mục 8.3](#83-cách-2--dev-local-không-docker-cloud-only-không-cần-gpu) |
| `1210 The temperature parameter is illegal.：限制小数点[2]位` | Retry của judge dùng `0.1 + 0.05 * attempt` → `0.15000000000000002`, provider Z.AI chỉ nhận 2 chữ số thập phân. Luôn rơi vào attempt 2 và 5 | Đã fix bằng `round(..., 2)` trong `modules/judge/multi_stage.py`. Nếu thấy lại, kiểm tra các chỗ khác truyền temperature tính bằng float |
| `Unparseable VLM response: ''` (judge) | Model trả content rỗng (thường do model reasoning trả hết vào reasoning field, hoặc provider lỗi) | Retry tự xử; nếu lặp lại liên tục thì đổi model judge |
| POST `/generate` trả 409 | Đang `generating` batch khác | Đợi `complete` hoặc restart nếu batch bị treo |

### Riêng trên Windows

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ValidationError: actors.critic.client='judge-critic-instance' points to a disabled client` | Chuyển sang OpenRouter nhưng mới đổi `coder`, quên `critic` (và `judge`) | Đổi `actors.critic.client` + `actors.judge.client` sang `"openrouter"` — bắt buộc kể cả khi `refinement_enabled: false` (xem [8.3 bước 3](#83-cách-2--dev-local-không-docker-cloud-only-không-cần-gpu)) |
| `UnicodeEncodeError: 'charmap' codec can't encode characters` (crash ở progress bar), hoặc log server in ra khối `Record was: {...}` | Console/redirect dùng code page `cp1252`, không encode được `→` `█` | `$env:PYTHONUTF8 = "1"` trước khi chạy `serve.py` và `test_pipeline.py` |
| `[Errno 10048] only one usage of each socket address ... bind on ('0.0.0.0', 10006)` | Lần chạy trước chưa chết hẳn, hoặc sidecar mồ côi còn giữ port | Dọn process (lệnh bên dưới) rồi chạy lại |
| Đã `Ctrl+C` nhưng RAM vẫn đầy, quạt vẫn quay | Windows không kill theo process group → `node.exe` sidecar + `chrome.exe` headless sống sót | Dọn process (lệnh bên dưới) |
| `curl : Cannot bind parameter 'Headers'...` hoặc `Invoke-WebRequest` báo lỗi tham số | `curl` trong PowerShell là alias của `Invoke-WebRequest` | Dùng `curl.exe` hoặc `Invoke-RestMethod` |
| Model trả 404 / `No endpoints found` | Vẫn để tên HF repo (`Tooony133/...`, `zai-org/...`) khi đã trỏ sang OpenRouter | Đổi sang model id có trên OpenRouter, cả 3 actor phải là model **vision** |
| `npm : File ...\npm.ps1 cannot be loaded` | ExecutionPolicy chặn script | `npm.cmd ci`, hoặc `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| `ModuleNotFoundError: config` | Chạy `python serve.py` từ gốc repo | `cd pipeline_service` trước khi chạy |
| Renderer fail hàng loạt / máy đứng | Mặc định `sidecar_count: 8` × `pool_size: 2` = 16 Chromium headless | Hạ `renderer.sidecar_count` xuống 1–2 khi dev trên laptop |

**Dọn process còn sót (PowerShell)** — chỉ giết Chromium của Puppeteer, không đụng Chrome cá nhân:

```powershell
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process chrome -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*\.cache\puppeteer\*" } | Stop-Process -Force
# còn kẹt port 10006 thì tìm và kill process đang giữ:
Get-NetTCPConnection -LocalPort 10006 -State Listen |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

**Công cụ debug chính:**

- `GET /debug/tasks` — tình trạng cả batch trong một màn hình: đang ở stage nào, điểm bao nhiêu, đã render chưa.
- `GET /debug/tasks/{stem}` — xem chi tiết 1 task: toàn bộ js_code, lỗi checker, ảnh render (base64), timing từng stage.
- `GET /debug/logs` hoặc file `pipeline_service/logs/pipeline.log` — log loguru đầy đủ, mỗi stage có tag riêng (`[MULTIGEN]`, `[BRACKET ...]`, `[TASK COMPLETED]`, `[Batch done]`…).
- Task fail luôn kèm `failure_stage` (prepare / coder / js_checker / renderer / critic / multigen / deadline) — biết ngay pipeline fail ở stage nào.
