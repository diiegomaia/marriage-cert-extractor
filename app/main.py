import os
import json
import torch
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from transformers import DonutProcessor, VisionEncoderDecoderModel
from huggingface_hub import snapshot_download
from pdf2image import convert_from_path
from PIL import Image

# ============================================================
# Configurações
# ============================================================
API_KEY     = os.environ.get("API_KEY")
HF_TOKEN    = os.environ.get("HF_TOKEN")
HF_REPO     = os.environ.get("HF_REPO")
MAX_LENGTH  = 1280
TASK_TOKEN  = "<s_certidao>"
MODELO_PATH = "/app/modelo"

app = FastAPI(title="Marriage Certificate Extractor")

# ============================================================
# Baixa o modelo apenas se o volume estiver vazio
# ============================================================
modelo_existe = Path(MODELO_PATH).exists() and any(Path(MODELO_PATH).iterdir())

if modelo_existe:
    print("✅ Modelo encontrado no volume. Carregando...")
else:
    print("⏳ Volume vazio. Baixando modelo do HuggingFace...")
    snapshot_download(
        repo_id=HF_REPO,
        token=HF_TOKEN,
        local_dir=MODELO_PATH
    )
    print("✅ Modelo baixado e salvo no volume!")

processor = DonutProcessor.from_pretrained(MODELO_PATH, local_files_only=True)
model     = VisionEncoderDecoderModel.from_pretrained(MODELO_PATH, local_files_only=True)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = model.to(device)
model.eval()
print(f"✅ Modelo carregado no dispositivo: {device}")


# ============================================================
# Funções auxiliares
# ============================================================
def carregar_imagem(caminho: str) -> Image.Image:
    if caminho.lower().endswith(".pdf"):
        paginas = convert_from_path(caminho, dpi=200)
        return paginas[0].convert("RGB")
    return Image.open(caminho).convert("RGB")


def extrair_dados(caminho: str) -> dict:
    imagem       = carregar_imagem(caminho)
    pixel_values = processor(imagem, return_tensors="pt").pixel_values.to(device)
    task_id      = processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=torch.tensor([[task_id]]).to(device),
            max_length=MAX_LENGTH,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )

    texto = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return {"raw": texto}


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def health_check():
    return {"status": "ok", "modelo": MODELO_PATH}


@app.post("/extrair")
async def extrair(
    file: UploadFile = File(...),
    x_api_key: str   = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida")

    if not file.filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Formato não suportado. Use PDF, JPG ou PNG")

    sufixo = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        dados = extrair_dados(tmp_path)
        return JSONResponse(content={
            "success": True,
            "data": dados
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )
    finally:
        os.unlink(tmp_path)