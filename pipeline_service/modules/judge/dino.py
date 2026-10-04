
from __future__ import annotations

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from logger_config import logger
from modules.judge.embedder_settings import EmbedderConfig


class DinoEmbedder:
    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._np: Any = None
        self._device: str | None = None
        self._disabled = False
        self._load_lock = asyncio.Lock()
        # Own executor: GPU-bound embeds would otherwise occupy asyncio's default
        # pool and stall every short to_thread job queued behind them.
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, config.workers), thread_name_prefix="dino"
        )

    async def _run(self, fn, *args) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _load_sync(self) -> bool:
        """Import deps + load the model on the calling thread. Returns success."""
        try:
            import numpy as np  
            import torch  
            from transformers import AutoImageProcessor, AutoModel  
        except Exception as exc:  
            logger.warning(f"[DINO] dependencies unavailable ({exc!r}); S2BV disabled")
            return False

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        from_pretrained_kwargs = {
            "revision": self.config.revision,
            "token": self.config.hf_token,
            "trust_remote_code": self.config.trust_remote_code,
        }
        try:
            processor = AutoImageProcessor.from_pretrained(
                self.config.model_id, **from_pretrained_kwargs
            )
            model = (
                AutoModel.from_pretrained(self.config.model_id, **from_pretrained_kwargs)
                .eval()
                .to(device)
            )
        except Exception as exc:  
            logger.warning(
                f"[DINO] failed to load model {self.config.model_id!r} on {device}: "
                f"{exc!r}; S2BV disabled"
            )
            return False

        self._np = np
        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device
        logger.info(f"[DINO] loaded {self.config.model_id} on {device}")
        return True

    async def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._disabled:
            return False
        async with self._load_lock:
            if self._model is not None:
                return True
            if self._disabled:
                return False
            ok = await self._run(self._load_sync)
            if not ok:
                self._disabled = True
            return ok

    def _embed_sync(self, images: list[bytes]) -> Any:
        """Return an (N, D) float32 L2-normalized numpy array for the given PNGs."""
        from PIL import Image  

        torch = self._torch
        pil = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        vecs = []
        bs = max(1, self.config.batch_size)
        for i in range(0, len(pil), bs):
            batch = pil[i : i + bs]
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model(**inputs)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]  
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vecs.append(pooled.detach().to("cpu").float().numpy())
        return self._np.concatenate(vecs, axis=0)

    @staticmethod
    def view_similarities(embeddings: bytes | None) -> dict[str, float]:
        """Cosine similarity of every `view_<name>` vector in a candidate npz to its `prompt` vector."""
        if not embeddings:
            return {}
        import numpy as np

        with np.load(io.BytesIO(embeddings)) as data:
            prompt = data["prompt"]
            return {
                key[len("view_"):]: round(float(np.dot(data[key], prompt)), 4)
                for key in data.files
                if key.startswith("view_")
            }

    @staticmethod
    def similarities(ref_vec: Any | None, vecs: dict[str, Any]) -> dict[str, float]:
        """Cosine similarity of in-memory view vectors to the reference vector."""
        if ref_vec is None or not vecs:
            return {}
        import numpy as np

        return {name: round(float(np.dot(v, ref_vec)), 4) for name, v in vecs.items()}

    async def embed_reference(self, image: bytes) -> Any | None:
        """Embed the reference image. Returns an opaque vector or None if disabled."""
        if not image or not await self._ensure_loaded():
            return None
        try:
            arr = await self._run(self._embed_sync, [image])
            return arr[0]
        except Exception as exc:  
            logger.warning(f"[DINO] reference embedding failed: {exc!r}")
            return None

    async def embed_views(self, views: dict[str, bytes]) -> dict[str, Any] | None:
        """Embed rendered views into in-memory vectors keyed by view name.

        Returns None when the embedder is disabled or nothing embeddable was given.
        """
        if not views or not await self._ensure_loaded():
            return None
        names = [n for n, b in views.items() if b]
        if not names:
            return None
        try:
            arr = await self._run(self._embed_sync, [views[n] for n in names])
        except Exception as exc:  
            logger.warning(f"[DINO] view embedding failed: {exc!r}")
            return None
        return dict(zip(names, arr))

    async def build_candidate_embeddings(
        self, ref_vec: Any | None, views: dict[str, bytes]
    ) -> bytes | None:
        """Embed candidate views and pack {prompt, view_<name>...} into npz bytes —
        the transport format the judge and the candidate export read.

        Returns None when the embedder is disabled, ``ref_vec`` is missing, or no
        views were rendered — the judge then runs S2BV-free.
        """
        if ref_vec is None:
            return None
        vecs = await self.embed_views(views)
        if not vecs:
            return None
        try:
            payload = {"prompt": ref_vec}
            for n, vec in vecs.items():
                payload[f"view_{n}"] = vec
            buf = io.BytesIO()
            self._np.savez(buf, **payload)
            return buf.getvalue()
        except Exception as exc:  
            logger.warning(f"[DINO] candidate embedding failed: {exc!r}")
            return None

    # ------------------------------------------------------------------ loc truoc bracket (DINOv3)
    # Chi so cua Minh Duong ("Idea: dung Dinov3 sang loc Candidate", 26/09): cosine CLS giua anh
    # de va render, CA HAI crop vuong quanh vat, 448 px; trung binh 2 goc TOT NHAT trong 4 goc
    # truoc cua S1. Do cua ho: giu winner 88% o k=20, dong thuan voi duel GLM tu tin 0,75.
    PREFILTER_VIEWS = ("front_left", "front_right", "front_below", "front_above")
    PREFILTER_SIZE = 448

    @staticmethod
    def _crop_square(img: Any, np: Any, margin: float = 0.08) -> Any:
        """Crop vuong quanh vat. Nen uoc tu vien anh (render nen trang, anh de nen xam studio)."""
        from PIL import Image

        a = np.asarray(img.convert("RGB"), dtype=np.int16)
        h, w, _ = a.shape
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg = np.median(border, axis=0)
        mask = (np.abs(a - bg).max(axis=2) > 18)
        ys, xs = np.nonzero(mask)
        if len(xs) < 16:
            return img.convert("RGB")
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        side = int(max(x1 - x0, y1 - y0) * (1 + 2 * margin)) + 1
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        left, top = cx - side // 2, cy - side // 2
        box = (max(0, left), max(0, top), min(w, left + side), min(h, top + side))
        out = Image.new("RGB", (side, side), tuple(int(c) for c in bg))
        out.paste(img.convert("RGB").crop(box), (box[0] - left, box[1] - top))  # phan ngoai anh = mau nen
        return out

    def _embed_crops_sync(self, images: list[bytes]) -> Any:
        """(N, D) L2-normalized CLS embeddings of object-centred square crops at 448 px."""
        from PIL import Image

        np, torch = self._np, self._torch
        size = self.PREFILTER_SIZE
        mean = np.array(getattr(self._processor, "image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(getattr(self._processor, "image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
        arrs = []
        for b in images:
            im = self._crop_square(Image.open(io.BytesIO(b)), np).resize((size, size), Image.BICUBIC)
            arrs.append(((np.asarray(im, dtype=np.float32) / 255.0 - mean) / std).transpose(2, 0, 1))
        vecs = []
        bs = max(1, self.config.batch_size)
        for i in range(0, len(arrs), bs):
            x = torch.from_numpy(np.stack(arrs[i : i + bs])).to(self._device)
            with torch.inference_mode():
                out = self._model(pixel_values=x)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]
            vecs.append(torch.nn.functional.normalize(pooled, dim=-1).float().cpu().numpy())
        return np.concatenate(vecs, axis=0)

    async def prefilter_scores(
        self, reference: bytes, views_by_k: dict[int, dict[str, bytes]], top: int = 2
    ) -> dict[int, float]:
        """Diem loc truoc cho tung ung vien: TB `top` goc tot nhat trong 4 goc truoc.

        Ung vien thieu goc truoc nao thi chi tinh tren cac goc co; khong co goc nao -> khong co diem.
        Tra {} khi DINO tat (worker khi do chay bracket day du nhu cu).
        """
        if not reference or not views_by_k or not await self._ensure_loaded():
            return {}
        order: list[tuple[int, str]] = []
        blobs: list[bytes] = [reference]
        for k, views in views_by_k.items():
            for name in self.PREFILTER_VIEWS:
                if views and views.get(name):
                    order.append((k, name))
                    blobs.append(views[name])
        if not order:
            return {}
        try:
            arr = await self._run(self._embed_crops_sync, blobs)
        except Exception as exc:
            logger.warning(f"[DINO] prefilter embedding failed: {exc!r}")
            return {}
        ref = arr[0]
        sims: dict[int, list[float]] = {}
        for (k, _), vec in zip(order, arr[1:]):
            sims.setdefault(k, []).append(float(vec @ ref))
        return {k: sum(sorted(v, reverse=True)[:top]) / min(top, len(v)) for k, v in sims.items()}
