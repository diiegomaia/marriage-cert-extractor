import os
import json
import torch
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from transformers import DonutProcessor, VisionEncoderDecoderModel
from huggingface_hub import hf_hub_download
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

ARQUIVOS_MODELO = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "historico.json",
]

processor = None
model     = None
device    = None

# ============================================================
# Startup — baixa o modelo se necessário e carrega
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, device

    modelo_path = Path(MODELO_PATH)
    tem_modelo  = (modelo_path / "model.safetensors").exists()
    tem_config  = (modelo_path / "processor_config.json").exists()

    # Log de diagnóstico
    arquivos = list(modelo_path.iterdir()) if modelo_path.exists() else []
    print(f"📁 Arquivos no volume: {[f.name for f in arquivos]}")
    print(f"   processor_config.json : {tem_config}")
    print(f"   model.safetensors     : {tem_modelo}")

    if not (tem_modelo and tem_config):
        print("⏳ Modelo incompleto ou ausente. Baixando arquivos individualmente...")
        modelo_path.mkdir(parents=True, exist_ok=True)

        for arquivo in ARQUIVOS_MODELO:
            destino = modelo_path / arquivo
            if destino.exists():
                print(f"   ⏭️  {arquivo} já existe, pulando...")
                continue
            print(f"   ⬇️  Baixando {arquivo}...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=arquivo,
                token=HF_TOKEN,
                local_dir=str(modelo_path)
            )
            print(f"   ✅ {arquivo} baixado!")

        print("✅ Todos os arquivos baixados!")
    else:
        print("✅ Modelo completo encontrado.")

    # Cria preprocessor_config.json no formato correto
    # O DonutProcessor exige este arquivo com image_processor_type na raiz
    preprocessor = modelo_path / "preprocessor_config.json"
    if not preprocessor.exists():
        preprocessor_content = {
            "data_format": "channels_first",
            "do_align_long_axis": True,
            "do_normalize": True,
            "do_pad": True,
            "do_rescale": True,
            "do_resize": True,
            "do_thumbnail": True,
            "image_mean": [0.5, 0.5, 0.5],
            "image_processor_type": "DonutImageProcessor",
            "image_std": [0.5, 0.5, 0.5],
            "resample": 2,
            "rescale_factor": 0.00392156862745098,
            "size": {"height": 1280, "width": 960}
        }
        with open(str(preprocessor), "w") as f:
            json.dump(preprocessor_content, f, indent=2)
        print("✅ preprocessor_config.json criado corretamente!")
    else:
        # Garante que o arquivo existente tem o formato correto
        with open(str(preprocessor), "r") as f:
            content = json.load(f)
        if "image_processor_type" not in content:
            content["image_processor_type"] = "DonutImageProcessor"
            with open(str(preprocessor), "w") as f:
                json.dump(content, f, indent=2)
            print("✅ preprocessor_config.json atualizado com image_processor_type!")

    print("⏳ Carregando modelo...")
    processor = DonutProcessor.from_pretrained(MODELO_PATH, local_files_only=True)
    model     = VisionEncoderDecoderModel.from_pretrained(MODELO_PATH, local_files_only=True)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Modelo carregado no dispositivo: {device}")

    yield

    print("🛑 Encerrando aplicação...")


app = FastAPI(title="Marriage Certificate Extractor", lifespan=lifespan)


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