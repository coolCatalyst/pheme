import io
import wave

import numpy as np
import uvicorn
from fastapi import Request, FastAPI
from fastapi.responses import StreamingResponse

from transformer_infer import PhemeClient

# Initialize Flask.
app = FastAPI()


class PhemeArgs:
    text = 'I gotta say, I would never expect that to happen!'
    manifest_path = 'demo/manifest.json'
    outputdir = 'demo'
    featuredir = 'demo/'
    text_tokens_file = 'ckpt/unique_text_tokens.k2symbols'
    t2s_path = 'ckpt/t2s/'
    s2a_path = 'ckpt/s2a/s2a.ckpt'
    target_sample_rate = 16000
    temperature = 0.7
    top_k = 210
    voice = 'male_voice'
    chunk_size = 100


model = PhemeClient(PhemeArgs())
file = open("audio.wav", "wb")
header_data = open("header_16000.raw", "rb").read()
file.write(header_data)

def postprocess(wav):
    """Post process the output waveform"""
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav


def encode_audio_common(
    frame_input, sample_rate=16000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


@app.post('/synthesize')
async def synthesize(request: Request):
    data = await request.json()
    text = data.pop("text")
    voice = data.pop("voice")

    async def stream_results():
        wavs = model.infer(text.replace('\n', ' '), voice=voice)
        for i, wav in enumerate(wavs):
            chunk = postprocess(wav)
            # if i == 0:
            #     yield encode_audio_common(b"")
            yield chunk.tobytes()
            file.write(chunk.tobytes())
        file.write(postprocess(np.zeros(16000, dtype=np.int16)).tobytes())
    return StreamingResponse(stream_results(), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7000,
        log_level="debug",
    )
