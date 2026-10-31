# Chạy thử worker ở local với prompts.txt

`local.py` chạy đúng hàm `generate()` mà runtime sẽ gọi. Nó chạy trên từng dòng của một
`prompts.txt` giống production, nhưng mọi thứ diễn ra trên máy bạn:

| Thành phần | Production | `local.py` |
| --- | --- | --- |
| Gọi model | qua witness, bằng subscription ChatGPT (Codex) hoặc API key | thẳng tới OpenAI, cũng bằng một trong hai cách đó |
| Render preview | renderer của Pod | renderer Docker chạy ở máy bạn |
| Kết quả | được nộp | chỉ lưu vào `outputs/local/`, **không nộp gì** |

Model call **vẫn dùng tài khoản thật**:
- chế độ API trừ tiền theo token;
- chế độ Codex trừ vào hạn mức subscription ChatGPT.

Mỗi call được lưu cache, nên chạy lại cùng prompt với code không đổi thì không tốn thêm.

## 1. Cài đặt (một lần)

Python 3.13, Node.js 20+ (cho checker) và Docker.

```sh
python3.13 -m venv .venv && . .venv/bin/activate
pip install openai==3.29.0 pydantic==2.14.1 numpy==2.5.4 pillow==13.0.0   # giống runtime
pip install ./path/to/renderer-client      # gói client renderer đi kèm SDK
```

Dựng renderer Three.js đã pin từ thư mục `render-service-js` của SDK, rồi chạy nó:

```sh
docker build --platform linux/amd64 -f render-service-js/Dockerfile.pinned \
  -t render-service-js:pinned render-service-js
export RENDERER_IMAGE_ID=$(docker image inspect render-service-js:pinned --format '{{.Id}}')
docker run -d --rm --init --platform linux/amd64 --cpus=2 --memory=2g \
  --name local-renderer -p 127.0.0.1:8000:8000 "$RENDERER_IMAGE_ID"
curl -sf http://127.0.0.1:8000/ping && echo ready
```

`--cpus=2 --memory=2g` giống giới hạn của renderer thật. Module quá nặng sẽ timeout ở đây giống như trên production.

## 2. Chạy

Chọn một trong hai chế độ, giống như khi chạy thật:

| Chế độ | Production (runner) | Local |
| --- | --- | --- |
| **Codex**: subscription ChatGPT (mặc định của runner) | không có `--api` | `--codex` |
| **API**: trả theo token | `--api` + `OPENAI_API_KEY` | `OPENAI_API_KEY`, không có `--codex` |

Nên test bằng đúng chế độ bạn sẽ chạy thật. Hai chế độ đi qua hai endpoint khác nhau
(`api.openai.com/v1/responses` và `chatgpt.com/backend-api/codex/responses`), và model mà
tài khoản được dùng có thể khác nhau.

### Chế độ API

```sh
export OPENAI_API_KEY=sk-...
python local.py --prompts prompts.txt --seed 1313431840 --limit 3
```

### Chế độ Codex (subscription ChatGPT)

Đăng nhập Codex CLI **một lần** bằng login dạng file. Witness cũng đọc đúng file này; login chỉ
lưu trong keyring thì không dùng được.

```sh
npm install -g @openai/codex        # hoặc cách cài khác của Codex CLI
codex -c 'cli_auth_credentials_store="file"' login
ls ~/.codex/auth.json               # phải có file này, auth_mode = "chatgpt"
```

Sau đó thêm `--codex`:

```sh
python local.py --prompts prompts.txt --seed 1313431840 --limit 3 --codex
```

- Script gọi `https://chatgpt.com/backend-api/codex/responses` giống witness. Request dùng token trong `auth.json` và header `ChatGPT-Account-Id`, luôn stream, có `store=false` và `tools=[]`. Kết quả lấy từ event `response.completed`, gồm cả `usage`.
- Khi token còn dưới 5 phút là hết hạn, script gọi `codex app-server` để làm mới (cần lệnh `codex` trong `PATH`). Nếu không làm mới được, hãy chạy lại `codex login`.
- Login nằm ở chỗ khác thì dùng `--codex-home DIR` (hoặc biến `CODEX_HOME`).
- Credit trong bảng tổng kết vẫn tính theo giá Standard, vì cuộc thi tính như vậy cho cả hai chế độ. Hoá đơn thật của subscription thì khác.
- Cache tách riêng theo chế độ: kết quả chạy bằng API không được dùng lại khi chạy `--codex`, và ngược lại.

Lần đầu hãy chạy `--limit 3` để xem chi phí và hạn mức, rồi mới chạy cả file.

| Tuỳ chọn | Ý nghĩa |
| --- | --- |
| `--prompts` | File mỗi dòng một URL ảnh hoặc đường dẫn file, đúng định dạng `prompts.txt` của round |
| `--seed` | Seed của round (hoặc bất kỳ số nào khi thử) |
| `--limit N` | Chỉ chạy N prompt đầu |
| `--only ID` | Chỉ chạy prompt có ID này; lặp lại được. ID là tên file ảnh, bỏ phần đuôi |
| `--parallel N` | Số prompt chạy cùng lúc (mặc định 4) |
| `--out DIR` | Thư mục kết quả (mặc định `outputs/local`, đã nằm trong `.gitignore`) |
| `--checker PATH` | Đường dẫn `tools/validate.js` của checker Three.js; chạy checker trên `final.js` |
| `--renderer-url` / `--renderer-id` | Mặc định đọc `RENDERER_URL` (hoặc `http://127.0.0.1:8000`) và `RENDERER_IMAGE_ID` |
| `--codex` | Dùng subscription ChatGPT qua login Codex thay cho `OPENAI_API_KEY` |
| `--codex-home DIR` | Thư mục login Codex (mặc định `CODEX_HOME` hoặc `~/.codex`) |

Kết thúc, script in bảng tổng kết (đây là output thật của một lần chạy thử):

```text
prompt                             status                 calls cached     credits renders     sec  checker
balloon                            ok                         7      0     990,000       5    42.3  PASSED
clock                              ok                         7      0     990,000       5    44.0  PASSED
pumpkin                            ok                         7      0     990,000       5    45.7  PASSED

3 prompts, 2,970,000 credits (≈ $2.97 at Standard prices)
```

Exit code là `0` khi mọi prompt `ok` và không prompt nào vượt 60 call / 5.000.000 credit; ngược lại là `1`.

## 3. Xem kết quả

```text
outputs/local/
  summary.json              # bảng tổng kết dạng JSON
  .cache/                   # response đã mua; xoá đi thì lần sau phải trả tiền lại
  <prompt_id>/
    image.png               # ảnh gốc đã tải
    final.js                # module mà generate() trả về
    final.png               # preview của final.js (để so bằng mắt với image.png)
    renders/01.js, 01.png   # mọi ứng viên đã render, theo thứ tự render
    calls.json              # từng call: request_id, model, token in/out, cached, text trả về
    result.json             # status, calls, credits, renders, thời gian
    checker.txt             # output của checker (nếu dùng --checker)
```

Cách đọc nhanh:

- **Chất lượng:** mở `final.png` cạnh `image.png`. Xem `renders/` để biết ứng viên nào thua và vì sao.
- **Chi phí:** cột `credits` của mỗi prompt phải nằm dưới 5.000.000 và có khoảng dư.
  - `calls.json` cho thấy call nào tốn nhiều nhất (thường là `output_tokens` của astra).
- **Lỗi:**
  - `status` là `error: ...` thì traceback nằm trong log ở terminal.
  - `None` nghĩa là `generate()` trả `None` (chỉ xảy ra khi đặt `placeholder=False`).
- **Log ở terminal** in từng call, ví dụ `call 11 gpt-6.1-sol | 5000 in / 8000 out | 2 calls, 330,000 credits`. Hai dòng cần chú ý:
  - `rejected:` là checker từ chối và đã gọi repair;
  - `budget admits k of n calls` là ngân sách đã cắt bớt một stage.

## 4. Vòng lặp chỉnh sửa

1. Sửa `src/config.py` (model, số draft, số vòng, token) hoặc `src/prompts.py`.
2. Chạy lại trên vài prompt khó, ví dụ `--only clock --only pumpkin`. Call nào không đổi
   request sẽ lấy từ cache (`cached` = `calls` nếu không có gì thay đổi).
3. So `final.png` và `credits` với lần trước. Nên đổi `--out` cho mỗi phiên bản để giữ kết quả cũ mà so sánh.
4. Khi hài lòng, chạy cả `prompts.txt`, commit, rồi mới chạy thật qua runner.

## Khác biệt với production

- Production gọi model qua witness, cùng endpoint và cùng cách xác thực như `local.py`. Tài khoản phải có quyền dùng các model trong `src/config.py` theo đúng chế độ bạn chọn.
- Cache ở local tương tự cách Pod tái sử dụng bản ghi: cùng nội dung request và cùng `request_id`.
  - Nếu chạy lại mà `cached` < `calls` dù không sửa gì, nghĩa là request không tất định. Production replay sẽ không khớp.
  - Kiểm tra lại xem có thời gian, `random` không seed hoặc thứ tự phụ thuộc thời điểm hay không.
- Thời gian chạy ở local không phản ánh giới hạn của round. Sandbox, giới hạn CPU/RAM của Python và bước kiểm replay chỉ có trên production.
