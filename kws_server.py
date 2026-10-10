"""SYNASPOT 流式关键词检测服务 (打断 / barge-in 接口).

基于 FastAPI 提供两类接口:

1. 离线注册 (HTTP REST, 生成关键词向量并落盘):
     POST   /api/keywords             注册 (Form: name + text 或 audio 文件)
     GET    /api/keywords             列出已注册关键词
     DELETE /api/keywords/{kid}       删除关键词

2. 在线流式检测 (WebSocket, 实时判断是否命中关键词):
     连接后发送 {"type":"start","keyword":"<id>"}
     持续发送 16kHz 单声道 int16 PCM 字节
     命中关键词时服务端推送 {"type":"interrupt", ...}
     期间持续推送 {"type":"score", ...} 供前端可视化

启动示例 (方案 A: SSL 主干, 支持文本+音频注册):
    python kws_server.py \
        --ckpt checkpoints/ssl/mixed/best.pt \
        --ssl-model-dir synaspot/Model_File/chinese-hubert-base \
        --threshold 0.5 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from kws_engine import KWSEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synaspot.kws_server")

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="SYNASPOT Keyword Spotting Server", version="1.0")

# 引擎在 main() 内初始化后注入
ENGINE: Optional[KWSEngine] = None
STATIC_INDEX = BASE_DIR / "static" / "index.html"


# ---------------------------------------------------------------- REST 注册
@app.get("/")
async def index():
    if STATIC_INDEX.exists():
        return FileResponse(str(STATIC_INDEX))
    return JSONResponse({"detail": "缺少 static/index.html"}, status_code=404)


@app.get("/api/keywords")
async def list_keywords():
    return {"keywords": ENGINE.list_keywords()}


@app.post("/api/keywords")
async def register_keyword(
    name: str = Form(...),
    text: str = Form(""),
    threshold: Optional[float] = Form(None),
    audio: Optional[UploadFile] = File(None),
):
    audio_path = None
    tmp = None
    try:
        if audio is not None:
            suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(await audio.read())
            tmp.close()
            audio_path = tmp.name
        meta = ENGINE.register(name=name.strip(), text=text.strip() or None,
                               audio_path=audio_path, threshold=threshold)
        return meta  # 静音裁剪与音频留档均由 engine.register 完成
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


@app.delete("/api/keywords/{kid}")
async def delete_keyword(kid: str):
    meta = ENGINE.delete_keyword(kid)
    if meta is None:
        return JSONResponse({"detail": f"关键词不存在: {kid}"}, status_code=404)
    return {"deleted": kid}


# ---------------------------------------------------------- WebSocket 检测
@app.websocket("/ws")
async def ws_detect(websocket: WebSocket):
    await websocket.accept()
    session = None  # DetectionSession
    kw_meta = None
    score_sent_at = 0.0
    session_pcm = bytearray()  # 累计本次会话收到的 PCM, 结束时落盘
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            data = msg.get("text") if msg.get("text") is not None else msg.get("bytes")

            if isinstance(data, str):
                try:
                    cmd = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error",
                                               "message": "非法JSON控制消息"})
                    continue
                ctype = cmd.get("type")
                if ctype == "start":
                    kid = cmd.get("keyword")
                    try:
                        kw_meta = (ENGINE.new_session(
                            kid,
                            window_sec=app.state.window_sec,
                            lookback_frames=app.state.lookback_frames,
                            cooldown_sec=app.state.cooldown_sec) if kid else None)
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                        continue
                    if kw_meta is None:
                        await websocket.send_json({"type": "error",
                                                   "message": "缺少 keyword"})
                        continue
                    session = kw_meta
                    session_pcm.clear()
                    # 支持连接时覆盖阈值
                    if cmd.get("threshold") is not None:
                        session.threshold = float(cmd["threshold"])
                    await websocket.send_json({
                        "type": "started",
                        "keyword": session.keyword["id"],
                        "name": session.keyword["name"],
                        "threshold": session.threshold,
                    })
                elif ctype == "stop":
                    await websocket.send_json({"type": "stopped"})
                    session = None
                    break
                elif ctype == "threshold":
                    if session is not None and cmd.get("value") is not None:
                        session.threshold = float(cmd["value"])
                        await websocket.send_json({"type": "threshold_set",
                                                   "threshold": session.threshold})
                continue

            if isinstance(data, bytes):
                if session is None:
                    await websocket.send_json({"type": "error",
                                               "message": "请先发送 start"})
                    continue
                session_pcm.extend(data)
                score, triggered = await asyncio.to_thread(session.feed_pcm, data)
                now = time.time()
                if triggered:
                    await websocket.send_json({
                        "type": "interrupt",
                        "keyword": session.keyword["id"],
                        "name": session.keyword["name"],
                        "score": round(score, 4),
                    })
                    score_sent_at = now
                elif now - score_sent_at >= 0.08:  # 节流 score 推送
                    await websocket.send_json({"type": "score",
                                               "score": round(score, 4)})
                    score_sent_at = now
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS 异常: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # 无论正常停止还是断开, 都把本次监听音频落盘, 便于离线测试
        if session_pcm:
            try:
                saved = save_session_audio(bytes(session_pcm), kw_meta)
                if saved:
                    logger.info(f"监听音频已保存: {saved['path']} ({saved['seconds']}s)")
                    try:
                        await websocket.send_json({"type": "saved",
                                                   "path": saved["path"],
                                                   "name": saved["name"],
                                                   "seconds": saved["seconds"]})
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"保存监听音频失败: {e}")


def save_session_audio(pcm: bytes, kw_meta) -> Optional[dict]:
    """把 int16 PCM 写成 16k 单声道 wav, 存到会话目录便于离线测试."""
    if not pcm:
        return None
    sess_dir = Path(app.state.session_dir)
    sess_dir.mkdir(parents=True, exist_ok=True)
    name = kw_meta.keyword["name"] if kw_meta is not None else "unknown"
    name = "".join(c for c in name if c not in r'\/:*?"<>|') or "unknown"
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{name}_{uuid.uuid4().hex[:4]}.wav"
    path = sess_dir / fname
    sr = ENGINE.sample_rate if ENGINE is not None else 16000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    seconds = round(len(pcm) / (2 * sr), 2)
    return {"path": str(path), "name": fname, "seconds": seconds}


# ------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description="SYNASPOT 流式关键词检测服务")
    ap.add_argument("--ckpt", required=True, help="SYNASPOT 检查点 (建议 ssl 主干以支持文本注册)")
    ap.add_argument("--ssl-model-dir", default=None,
                    help="Chinese HuBERT 目录 (ssl 主干时需要)")
    ap.add_argument("--store-dir", default=str(BASE_DIR / "keyword_store"),
                    help="关键词向量存储目录")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="默认命中阈值 (注册时可按关键词覆盖)")
    ap.add_argument("--window-sec", type=float, default=None,
                    help="检测滑窗秒数, 默认自适应为注册音频时长")
    ap.add_argument("--lookback-frames", type=int, default=None,
                    help="峰值帧回看数, 默认取整个窗口")
    ap.add_argument("--cooldown-sec", type=float, default=2.0, help="触发冷却秒数")
    ap.add_argument("--session-dir", default=str(BASE_DIR / "test_audio"),
                    help="监听会话音频保存目录 (便于离线测试)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    return ap.parse_args()


def main():
    global ENGINE
    args = parse_args()
    ENGINE = KWSEngine(args.ckpt, device=args.device, store_dir=args.store_dir,
                       default_threshold=args.threshold, ssl_model_dir=args.ssl_model_dir)
    app.state.engine = ENGINE
    # 供检测会话复用滑窗参数
    app.state.window_sec = args.window_sec
    app.state.lookback_frames = args.lookback_frames
    app.state.cooldown_sec = args.cooldown_sec
    app.state.session_dir = args.session_dir
    logger.info(f"服务启动: http://{args.host}:{args.port}")
    logger.info(f"关键词存储目录: {args.store_dir}")
    logger.info(f"监听音频保存目录: {args.session_dir}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()