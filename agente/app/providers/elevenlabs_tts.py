"""
Provider: ElevenLabs TTS

Converte texto editorial em áudio MP3 via ElevenLabs API.
Gera dois arquivos por run:
  - <run_id>_audio_principal.mp3  ← TEXTO PRINCIPAL (pago)
  - <run_id>_audio_gratuito.mp3   ← TEXTO GRATUITO (público)

Configuração:
  ELEVENLABS_API_KEY=sk_...
  ELEVENLABS_VOICE_ID=XrMNSxvVxLkUlaSeEuLM  (padrão: rafael real)
"""

from __future__ import annotations

import re
from pathlib import Path

from app.audit.logger import get_logger
from app.config.settings import settings

_log = get_logger("providers.elevenlabs_tts")

# Seção principal por modo
_PRINCIPAL_SECTION = {
    "week_recap": "WEEK RECAP",
    "week_ahead": "TEXTO PRINCIPAL",
    "growth": "TEXTO PRINCIPAL",
    "flow_show": "TEXTO PRINCIPAL",
    "tese": "TEXTO PRINCIPAL",
    "tese_livre": "TEXTO PRINCIPAL",
    "morning_call": "TEXTO PRINCIPAL",
    "podcast_sabado": "SCRIPT PODCAST",
}

_GRATUITO_SECTION = "TEXTO GRATUITO"


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_audio(text: str, mode: str, out_dir: Path, run_id: str) -> dict[str, str]:
    """
    Gera áudios MP3 do TEXTO PRINCIPAL e TEXTO GRATUITO.
    Retorna dict com keys 'audio_principal' e/ou 'audio_gratuito' → paths.
    """
    key = settings.elevenlabs_api_key
    if not key:
        _log.warning("elevenlabs_key_missing")
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    principal_section = _PRINCIPAL_SECTION.get(mode, "TEXTO PRINCIPAL")

    # TEXTO PRINCIPAL / WEEK RECAP / SCRIPT PODCAST
    principal_text = _extract_section(text, principal_section)
    if principal_text:
        raw_path = out_dir / f"{run_id}_audio_principal_raw.mp3"
        ok = _synthesize(principal_text, raw_path, key)
        if ok:
            # Podcast de sábado: concatena intro + locução + outro
            if mode == "podcast_sabado":
                final_path = out_dir / f"{run_id}_audio_principal.mp3"
                _concat_podcast(raw_path, final_path)
                raw_path.unlink(missing_ok=True)
                results["audio_principal"] = str(final_path)
            else:
                final_path = out_dir / f"{run_id}_audio_principal.mp3"
                if final_path.exists():
                    final_path.unlink()
                raw_path.rename(final_path)
                results["audio_principal"] = str(final_path)
            _log.info("tts_principal_done", chars=len(principal_text),
                      path=results["audio_principal"].split("\\")[-1])

    # TEXTO GRATUITO
    gratuito_text = _extract_section(text, _GRATUITO_SECTION)
    if gratuito_text:
        path = out_dir / f"{run_id}_audio_gratuito.mp3"
        ok = _synthesize(gratuito_text, path, key)
        if ok:
            results["audio_gratuito"] = str(path)
            _log.info("tts_gratuito_done", chars=len(gratuito_text), path=path.name)

    return results


# ── Extração de seção ──────────────────────────────────────────────────────────

def _extract_section(text: str, section_name: str) -> str:
    """Extrai o conteúdo de uma seção delimitada por === NOME ===."""
    # Encontra o início da seção
    pattern = re.compile(
        r"={3}\s*" + re.escape(section_name) + r"\s*={3}(.*?)(?:={3}|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(text)
    if not m:
        return ""
    raw = m.group(1).strip()
    return _clean_for_tts(raw)


def _clean_for_tts(text: str) -> str:
    """
    Prepara o texto para locução natural em português do Brasil.
    Remove artefatos visuais, expande siglas problemáticas e garante
    fluidez oral — evita roboticidade no ElevenLabs.
    """
    # Remove sentinels <<<IMG:...>>>
    text = re.sub(r"<<<IMG:[^>]*>>>", "", text)
    # Remove separadores ━━━ e ---
    text = re.sub(r"[━─]{3,}", "\n", text)
    # Remove linhas de marcador de seção (=== ... ===)
    text = re.sub(r"={3}[^=\n]+={3}", "", text)
    # Remove markdown: **bold**, *italic*, ### headings
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    # Remove traços de lista (- item) no início de linha
    text = re.sub(r"^\s*[-–—]\s+", "", text, flags=re.MULTILINE)

    # ── Números: manter como dígitos para o ElevenLabs ler naturalmente ────────
    # Converte "R$ 1.234,56" para "1234 reais" (evita engasgos)
    text = re.sub(r"R\$\s*([\d.,]+)", lambda m: m.group(1).replace(".", "").replace(",", ".") + " reais", text)
    # Converte "$110" para "110 dólares"
    text = re.sub(r"\$([\d.,]+)", lambda m: m.group(1).replace(",", "") + " dólares", text)
    # Converte "110%" para "110 por cento"
    text = re.sub(r"([\d]+(?:[.,]\d+)?)\s*%", r"\1 por cento", text)
    # Remove vírgulas de milhar em números (1,234 → 1234)
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)

    # ── Expansão de siglas para locução natural ────────────────────────────────
    _SIGLAS = [
        # Palavras com acentuação difícil para TTS
        (r"\breprecifica[rção]*\b",          "ajusta o preço"),
        (r"\breprecificação\b",              "ajuste de preço"),
        (r"\bprecifica[rção]*\b",            "precifica"),
        (r"\bUSA\b",                        "Estados Unidos da América"),
        (r"\bEUA\b",                        "Estados Unidos"),
        (r"\bFed\b",                        "Federal Reserve"),
        (r"\bFOMC\b",                       "Comitê de Mercado Aberto do Federal Reserve"),
        (r"\bGDP\b",                        "produto interno bruto"),
        (r"\bPIB\b",                        "produto interno bruto"),
        (r"\bCPI\b",                        "índice de preços ao consumidor"),
        (r"\bPPI\b",                        "índice de preços ao produtor"),
        (r"\bPCE\b",                        "índice de gastos com consumo pessoal"),
        (r"\bNFP\b",                        "folha de pagamentos não-agrícola"),
        (r"\bECB\b",                        "Banco Central Europeu"),
        (r"\bBCE\b",                        "Banco Central Europeu"),
        (r"\bETFs\b",                       "fundos de índice"),
        (r"\bETF\b",                        "fundo de índice"),
        (r"\bS&P\b",                        "S e P"),
        (r"\bSPX\b",                        "S e P 500"),
        (r"\bVIX\b",                        "índice de volatilidade"),
        (r"\bGNL\b",                        "gás natural liquefeito"),
        (r"\bLNG\b",                        "gás natural liquefeito"),
        (r"\bYTD\b",                        "no acumulado do ano"),
        (r"\bQoQ\b",                        "em relação ao trimestre anterior"),
        (r"\bYoY\b",                        "em relação ao mesmo período do ano anterior"),
        (r"\bMoM\b",                        "em relação ao mês anterior"),
        (r"\bbps\b",                        "pontos base"),
        (r"\bPM\b",                         "primeiro-ministro"),
        (r"\bCEO\b",                        "diretor executivo"),
        (r"\bCFO\b",                        "diretor financeiro"),
        (r"\bCOO\b",                        "diretor de operações"),
        (r"\bCTO\b",                        "diretor de tecnologia"),
        (r"\bISM\b",                        "Instituto de Gestão de Suprimentos"),
        (r"\bJOLTS\b",                      "pesquisa de vagas e rotatividade de empregos"),
        (r"\bFAANG\b",                      "grandes empresas de tecnologia"),
        (r"\bMagnificent\s+Sev[ae]n\b",     "as sete gigantes de tecnologia"),
        (r"\bMagnificent\s+7\b",            "as sete gigantes de tecnologia"),
        (r"\bMag\s*7\b",                    "as sete gigantes de tecnologia"),
        (r"\bMag\s*Seven\b",                "as sete gigantes de tecnologia"),
        (r"\bWall\s+Street\b",              "Uôl Estrit"),
        (r"\bMain\s+Street\b",              "a economia real"),
        (r"\bdeep\s*fake[s]?\b",            "falsificação digital"),
        (r"\bdeep\s*dive\b",                "análise aprofundada"),
        (r"\bbullish\b",                    "otimista"),
        (r"\bbearish\b",                    "pessimista"),
        (r"\brally\b",                      "recuperação"),
        (r"\bselloff\b",                    "queda acentuada"),
        (r"\bsell[\s-]off\b",               "queda acentuada"),
        (r"\bdefault\b",                    "calote"),
        (r"\bspread[s]?\b",                 "spread"),
        (r"\byield[s]?\b",                  "rendimento"),
        (r"\bhawkish\b",                    "linha dura"),
        (r"\bdovish\b",                     "linha branda"),
        (r"\bquantitative\s+easing\b",      "afrouxamento quantitativo"),
        (r"\bquantitative\s+tightening\b",  "aperto quantitativo"),
        (r"\bAI\b",                         "inteligência artificial"),
        (r"\bIPO[s]?\b",                    "oferta inicial de ações"),
        (r"\bM&A\b",                        "fusões e aquisições"),
        (r"\bGPU[s]?\b",                    "unidade de processamento gráfico"),
        (r"\bNATO\b",                       "Organização do Tratado do Atlântico Norte"),
        (r"\bOPEC\b",                       "Organização dos Países Exportadores de Petróleo"),
        (r"\bIMF\b",                        "Fundo Monetário Internacional"),
        (r"\bFMI\b",                        "Fundo Monetário Internacional"),
        (r"\bWTO\b",                        "Organização Mundial do Comércio"),
        (r"\bOMC\b",                        "Organização Mundial do Comércio"),
    ]
    for pattern, replacement in _SIGLAS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # ── Tradução de termos em inglês que TTS pronunciaria errado em PT ─────────
    # Respeita case quando possível
    _TRADUCOES_EN_PT = [
        (r"\bbase metals\b",          "metais base"),
        (r"\bprecious metals\b",      "metais preciosos"),
        (r"\bmetals\b",               "metais"),
        (r"\bcommodities\b",          "commodities"),  # já é usado em PT
        (r"\bcommodity\b",            "commodity"),
        (r"\boil and gas\b",          "óleo e gás"),
        (r"\bpayroll\b",              "folha de pagamento"),
        (r"\bhedge funds?\b",         "fundos hedge"),
        (r"\bmarket makers?\b",       "formadores de mercado"),
        (r"\bdealers?\b",             "dealers"),  # jargão mantido
        (r"\bshortfall\b",            "déficit"),
        (r"\bdrawdown\b",             "queda"),
        (r"\bbreakdown\b",            "colapso"),
        (r"\bbreakout\b",             "rompimento"),
        (r"\bguidance\b",             "projeção"),
        (r"\bbeat\b",                 "superou"),
        (r"\bmiss\b",                 "decepcionou"),
        (r"\bguidance cut\b",         "corte de projeção"),
        (r"\bsemiconductors?\b",      "semicondutores"),
    ]
    for pattern, replacement in _TRADUCOES_EN_PT:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove linhas em branco excessivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Concatenação intro + locução + outro ──────────────────────────────────────

def _concat_podcast(tts_path: Path, dest: Path) -> None:
    """
    Junta intro + locução gerada + outro em um único arquivo MP3.
    Normaliza volume entre os segmentos para equilíbrio de áudio.
    Tenta: ffmpeg direto (Python 3.14+) > pydub > fallback bytes.
    """
    intro_path = settings.podcast_intro_path
    outro_path = settings.podcast_outro_path

    # Tenta ffmpeg direto primeiro (funciona em Python 3.14 sem audioop)
    try:
        _concat_podcast_ffmpeg(tts_path, dest, intro_path, outro_path)
        return
    except Exception as exc:
        _log.warning("podcast_ffmpeg_failed", error=str(exc)[:150])

    try:
        _concat_podcast_pydub(tts_path, dest, intro_path, outro_path)
        return
    except Exception as exc:
        _log.warning("podcast_pydub_failed_fallback_bytes", error=str(exc)[:150])

    _concat_podcast_bytes(tts_path, dest, intro_path, outro_path)


def _concat_podcast_ffmpeg(tts_path: Path, dest: Path,
                            intro_path: Path, outro_path: Path) -> None:
    """
    Concat via ffmpeg direto (subprocess). Funciona em Python 3.14+ (sem audioop).
    Normaliza volume via filtro loudnorm.
    """
    import subprocess
    import tempfile
    import static_ffmpeg
    static_ffmpeg.add_paths()

    # Monta lista de arquivos na ordem correta
    parts: list[Path] = []
    if intro_path.exists():
        parts.append(intro_path)
        _log.info("podcast_intro_added", path=intro_path.name, kb=intro_path.stat().st_size // 1024)
    else:
        _log.warning("podcast_intro_missing", path=str(intro_path))

    parts.append(tts_path)

    if outro_path.exists():
        parts.append(outro_path)
        _log.info("podcast_outro_added", path=outro_path.name, kb=outro_path.stat().st_size // 1024)
    else:
        _log.warning("podcast_outro_missing", path=str(outro_path))

    if not parts:
        raise RuntimeError("nenhum segmento de audio disponivel")

    # Cria arquivo de lista para ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as lf:
        for p in parts:
            # ffmpeg concat demuxer exige escape de barras invertidas e aspas
            p_str = str(p.absolute()).replace("\\", "/").replace("'", r"\'")
            lf.write(f"file '{p_str}'\n")
        list_path = Path(lf.name)

    try:
        # Primeiro tenta concat direto (copy sem re-encoding)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            # Se copy falhou (formatos diferentes), re-encoda para MP3
            _log.info("podcast_concat_reencode", reason=result.stderr[:100])
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c:a", "libmp3lame", "-b:a", "128k",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                str(dest),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg re-encode failed: {result.stderr[:200]}")

        total_kb = dest.stat().st_size // 1024
        _log.info("podcast_concat_done_ffmpeg", path=dest.name, total_kb=total_kb, parts=len(parts))
    finally:
        list_path.unlink(missing_ok=True)


def _concat_podcast_pydub(tts_path: Path, dest: Path,
                           intro_path: Path, outro_path: Path) -> None:
    """Concatenação com normalização de volume via pydub."""
    import static_ffmpeg
    static_ffmpeg.add_paths()

    from pydub import AudioSegment
    from pydub.effects import normalize

    TARGET_DBFS = -16.0  # alvo de volume normalizado

    def _load_and_normalize(path: Path) -> AudioSegment | None:
        if not path.exists():
            return None
        seg = AudioSegment.from_file(str(path))
        # Normaliza para target, depois aplica ajuste fino
        normalized = normalize(seg)
        change = TARGET_DBFS - normalized.dBFS
        return normalized.apply_gain(change)

    tts_seg = AudioSegment.from_file(str(tts_path))
    tts_norm = normalize(tts_seg)
    tts_norm = tts_norm.apply_gain(TARGET_DBFS - tts_norm.dBFS)

    parts: list[AudioSegment] = []

    intro_seg = _load_and_normalize(intro_path)
    if intro_seg is not None:
        parts.append(intro_seg)
        _log.info("podcast_intro_added", path=intro_path.name, dbfs=round(intro_seg.dBFS, 1))
    else:
        _log.warning("podcast_intro_missing", path=str(intro_path))

    parts.append(tts_norm)

    outro_seg = _load_and_normalize(outro_path)
    if outro_seg is not None:
        parts.append(outro_seg)
        _log.info("podcast_outro_added", path=outro_path.name, dbfs=round(outro_seg.dBFS, 1))
    else:
        _log.warning("podcast_outro_missing", path=str(outro_path))

    combined = parts[0]
    for part in parts[1:]:
        combined = combined + part

    combined.export(str(dest), format="mp3", bitrate="128k")
    _log.info("podcast_concat_done", path=dest.name, total_kb=len(dest.read_bytes()) // 1024)


def _concat_podcast_bytes(tts_path: Path, dest: Path,
                           intro_path: Path, outro_path: Path) -> None:
    """Fallback: concatenação direta de bytes sem normalização."""
    chunks: list[bytes] = []
    total_kb = 0

    if intro_path.exists():
        data = intro_path.read_bytes()
        chunks.append(data)
        total_kb += len(data) // 1024
        _log.info("podcast_intro_added", path=intro_path.name, kb=len(data) // 1024)
    else:
        _log.warning("podcast_intro_missing", path=str(intro_path))

    tts_data = tts_path.read_bytes()
    chunks.append(tts_data)
    total_kb += len(tts_data) // 1024

    if outro_path.exists():
        data = outro_path.read_bytes()
        chunks.append(data)
        total_kb += len(data) // 1024
        _log.info("podcast_outro_added", path=outro_path.name, kb=len(data) // 1024)
    else:
        _log.warning("podcast_outro_missing", path=str(outro_path))

    dest.write_bytes(b"".join(chunks))
    _log.info("podcast_concat_done_bytes", path=dest.name, total_kb=total_kb)


def _join_chunks_ffmpeg(chunk_paths: list[Path], dest: Path) -> None:
    """
    Junta chunks TTS via ffmpeg (Python 3.14 compatible, sem pydub).
    Usa concat demuxer com re-encoding para garantir que todos os chunks
    sejam preservados (byte concat de mp3s com headers diferentes corta chunks).
    Adiciona 800ms de silêncio no final para evitar truncation pelo encoder.
    """
    import subprocess
    import tempfile
    import static_ffmpeg
    static_ffmpeg.add_paths()

    if len(chunk_paths) == 1:
        # Um único chunk — só adiciona silêncio no final via ffmpeg
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(chunk_paths[0]),
            "-af", "apad=pad_dur=0.8",
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # Fallback: copy sem padding
            dest.write_bytes(chunk_paths[0].read_bytes())
        return

    # Múltiplos chunks: concat demuxer com re-encoding
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as lf:
        for p in chunk_paths:
            p_str = str(p.absolute()).replace("\\", "/").replace("'", r"\'")
            lf.write(f"file '{p_str}'\n")
        list_path = Path(lf.name)

    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-af", "apad=pad_dur=0.8",  # 800ms silence end (evita truncate)
            "-c:a", "libmp3lame", "-b:a", "128k",
            str(dest),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            _log.warning("tts_join_ffmpeg_failed", err=result.stderr[:200])
            # Fallback bytes (quebrado mas último recurso)
            dest.write_bytes(b"".join(p.read_bytes() for p in chunk_paths))
        else:
            _log.info("tts_joined_ffmpeg", chunks=len(chunk_paths), bytes=dest.stat().st_size)
    finally:
        list_path.unlink(missing_ok=True)


# ── Síntese via ElevenLabs SDK ────────────────────────────────────────────────

_ELEVENLABS_CHAR_LIMIT = 4800  # margem de segurança abaixo do limite de 5000


def _split_text(text: str, limit: int = _ELEVENLABS_CHAR_LIMIT) -> list[str]:
    """Divide texto em chunks respeitando parágrafos e o limite de caracteres."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        # Se parágrafo sozinho já ultrapassa, divide por frase
        if len(para) > limit:
            for sentence in re.split(r"(?<=[.!?])\s+", para):
                if len(current) + len(sentence) + 2 > limit:
                    if current:
                        chunks.append(current.strip())
                    current = sentence
                else:
                    current = (current + " " + sentence).strip() if current else sentence
        elif len(current) + len(para) + 2 > limit:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current:
        chunks.append(current.strip())
    return chunks


def _synthesize(text: str, dest: Path, api_key: str) -> bool:
    """Chama ElevenLabs TTS e salva MP3. Divide em chunks se necessário. Retorna True se ok."""
    try:
        from elevenlabs.client import ElevenLabs
        from elevenlabs import VoiceSettings

        client = ElevenLabs(api_key=api_key)
        voice_id = settings.elevenlabs_voice_id
        voice_settings = VoiceSettings(
            stability=0.30,
            similarity_boost=1.0,
            style=0.49,
            use_speaker_boost=True,
            speed=1.15,
        )

        # Frase de fechamento audível — "..." sozinho é ignorado pelo ElevenLabs
        text_with_ending = text.rstrip() + "\n\n. . ."

        chunks = _split_text(text_with_ending)
        _log.info("tts_chunks", total=len(chunks), chars=len(text))

        chunk_paths: list[Path] = []
        for i, chunk in enumerate(chunks):
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=chunk,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings=voice_settings,
            )
            # Consumir generator completamente via list() antes de join
            if hasattr(audio, "__iter__") and not isinstance(audio, (bytes, bytearray)):
                chunk_bytes = b"".join(list(audio))
            else:
                chunk_bytes = bytes(audio)

            chunk_path = dest.parent / f"{dest.stem}_chunk{i}.mp3"
            chunk_path.write_bytes(chunk_bytes)
            chunk_paths.append(chunk_path)
            _log.info("tts_chunk_done", chunk=i + 1, total=len(chunks), bytes=len(chunk_bytes))

        # Join via ffmpeg subprocess (Python 3.14 compatible, preserva todos os chunks)
        _join_chunks_ffmpeg(chunk_paths, dest)

        for cp in chunk_paths:
            cp.unlink(missing_ok=True)

        _log.info("tts_saved", bytes=dest.stat().st_size, path=dest.name)
        return True

    except Exception as exc:
        _log.warning("tts_error", error=str(exc))
        return False
