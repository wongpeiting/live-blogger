#!/usr/bin/env python3
"""
Live Pipeline: YouTube → Transcription → Auto Live Blog (Threaded)

Threaded architecture where listening NEVER stops:
  - Listener thread: yt-dlp → ffmpeg → Whisper → transcript cleanup → JSONL
  - Coordinator (main thread): watches transcript, detects triggers, dispatches
  - Blogger pool (ThreadPoolExecutor, max 3): Gemini draft → editor pass → liveblog.md

Usage:
    python3 live-pipeline.py <youtube-url> [--chunk-seconds 30] [--max-bloggers 3]

Requirements:
    yt-dlp, ffmpeg, transformers, torch, google-genai, pyannote.audio
Environment:
    GOOGLE_API_KEY, HF_TOKEN
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import re

import torch
from transformers import pipeline as hf_pipeline
from google import genai


# ─── Corrections Cache (applies corrections.json to chunk text) ──────────────

CORRECTIONS_FILE = "corrections.json"


class CorrectionCache:
    """Loads corrections.json and applies name/term/abbreviation fixes to text.

    Reloads automatically when the file changes, just like fix-transcript.py.
    Used by the coordinator to clean chunks before triage and blogging.
    """

    def __init__(self, path: str = CORRECTIONS_FILE):
        self.path = path
        self._corrections: dict = {}
        self._mtime: float = 0

    def _reload_if_changed(self):
        if not os.path.exists(self.path):
            return
        mtime = os.path.getmtime(self.path)
        if mtime != self._mtime:
            self._mtime = mtime
            with open(self.path) as f:
                self._corrections = json.load(f)

    def apply(self, text: str) -> str:
        """Apply all corrections to text."""
        self._reload_if_changed()
        if not self._corrections:
            return text
        # Names (case-sensitive, longest first)
        for wrong, right in sorted(
            self._corrections.get("names", {}).items(), key=lambda x: -len(x[0])
        ):
            text = text.replace(wrong, right)
        # Terms (case-insensitive, longest first)
        for wrong, right in sorted(
            self._corrections.get("terms", {}).items(), key=lambda x: -len(x[0])
        ):
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
        # Abbreviations (word-boundary, longest first)
        for wrong, right in sorted(
            self._corrections.get("abbreviations", {}).items(), key=lambda x: -len(x[0])
        ):
            text = re.sub(r"\b" + re.escape(wrong) + r"\b", right, text)
        return text


# ─── Prompts (loaded from external config) ────────────────────────────────────

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
PROMPTS_CONFIG = os.path.join(PROMPTS_DIR, "config.json")


def _load_prompts() -> dict:
    """Load prompt configuration from prompts/config.json."""
    if not os.path.exists(PROMPTS_CONFIG):
        print(f"[error] Prompts config not found: {PROMPTS_CONFIG}")
        print("        Copy prompts/config.example.json to prompts/config.json and customise.")
        sys.exit(1)
    with open(PROMPTS_CONFIG) as f:
        return json.load(f)


_PROMPTS = _load_prompts()

LIVE_BLOG_SYSTEM_PROMPT = _PROMPTS["live_blog_system_prompt"]
EDITOR_SYSTEM_PROMPT = _PROMPTS["editor_system_prompt"]
LIVE_BLOG_USER_PROMPT = _PROMPTS["live_blog_user_prompt"]
EDITOR_USER_PROMPT = _PROMPTS["editor_user_prompt"]
TRIAGE_SYSTEM_PROMPT = _PROMPTS["triage_system_prompt"]
TRIAGE_USER_PROMPT = _PROMPTS["triage_user_prompt"]
_SPEAKER_TYPE_LABELS = _PROMPTS.get("speaker_type_labels", {})


# ─── Transcript Cleanup ──────────────────────────────────────────────────────

TERM_CORRECTIONS = {
    # MP name corrections (common Whisper garbles)
    "Wirtz Chia": "Edward Chia",
    "Wirt Chia": "Edward Chia",
    # Domain terms
    "grit stability": "grid stability",
    "L N G": "LNG",
    "L.N.G.": "LNG",
    "M T I": "MTI",
    "M.T.I.": "MTI",
    "Hormos": "Hormuz",
    "A I": "AI",
    "A.I.": "AI",
    "Beps": "BEPS",
    "De Carbonization": "decarbonisation",
    "de carbonization": "decarbonisation",
    "Electrocity": "electricity",
    "electrocity": "electricity",
    "bio diversity": "biodiversity",
    "cyber security": "cybersecurity",
    "block chain": "blockchain",
    "E V": "EV",
    "E.V.": "EV",
}

ABBREVIATION_CORRECTIONS = {
    "C P F": "CPF",
    "H D B": "HDB",
    "G S T": "GST",
    "N S": "NS",
    "M O E": "MOE",
    "M O H": "MOH",
    "M O F": "MOF",
    "M N D": "MND",
    "M H A": "MHA",
    "M O T": "MOT",
    "L T A": "LTA",
    "E D B": "EDB",
    "J T C": "JTC",
    "N E A": "NEA",
    "P U B": "PUB",
    "S P F": "SPF",
    "S A F": "SAF",
    "M O M": "MOM",
    "N D P": "NDP",
    "S M E": "SME",
    "S M Es": "SMEs",
    "G R C": "GRC",
    "S M C": "SMC",
    "N C M P": "NCMP",
    "N M P": "NMP",
    "B T O": "BTO",
    "C O E": "COE",
    "P S L E": "PSLE",
    "I S D": "ISD",
    "M A S": "MAS",
    "P A P": "PAP",
    "W P": "WP",
    "P S P": "PSP",
}


class MPIndex:
    """Phonetic index for fuzzy matching MP names from Whisper transcripts."""

    def __init__(self, mp_list_path: str):
        self.mps = []
        self.name_tokens = {}  # lowercase token → list of MP indices
        if os.path.exists(mp_list_path):
            with open(mp_list_path) as f:
                self.mps = json.load(f)
            self._build_index()

    def _build_index(self):
        for i, mp in enumerate(self.mps):
            name = mp["name"]
            tokens = name.lower().split()
            for token in tokens:
                self.name_tokens.setdefault(token, []).append(i)

    def find_closest(self, garbled_name: str, threshold: float = 0.7) -> dict | None:
        """Find the closest MP to a garbled transcription. Returns full MP dict or None."""
        garbled_lower = garbled_name.lower().strip()
        garbled_tokens = garbled_lower.split()

        best_score = 0.0
        best_mp = None

        for mp in self.mps:
            mp_name_lower = mp["name"].lower()
            # Full name similarity
            score = SequenceMatcher(None, garbled_lower, mp_name_lower).ratio()
            if score > best_score:
                best_score = score
                best_mp = mp

            # Token-level matching (last name is most reliable)
            mp_tokens = mp_name_lower.split()
            if len(garbled_tokens) >= 1 and len(mp_tokens) >= 1:
                # Compare last tokens (surname)
                last_score = SequenceMatcher(
                    None, garbled_tokens[-1], mp_tokens[-1]
                ).ratio()
                if last_score > 0.8:
                    # Surname match — check first name loosely
                    if len(garbled_tokens) >= 2 and len(mp_tokens) >= 2:
                        first_score = SequenceMatcher(
                            None, garbled_tokens[0], mp_tokens[0]
                        ).ratio()
                        combined = (last_score * 0.6) + (first_score * 0.4)
                    else:
                        combined = last_score * 0.8
                    if combined > best_score:
                        best_score = combined
                        best_mp = mp

        if best_score >= threshold and best_mp:
            return best_mp
        return None

    def find_closest_name(self, garbled_name: str, threshold: float = 0.7) -> str | None:
        """Find closest MP name (string only). Convenience wrapper."""
        mp = self.find_closest(garbled_name, threshold)
        return mp["name"] if mp else None

    @staticmethod
    def speaker_type(mp: dict) -> str:
        """Classify an MP into one of four speaker types.

        Returns: 'cabinet', 'pap_backbench', 'opposition', 'nmp'
        """
        title = mp.get("title", "")
        party = mp.get("party", "")

        # NMPs
        if party == "NMP" or "Nominated" in title:
            return "nmp"

        # Opposition (WP elected + NCMPs, PSP NCMPs)
        if party == "WP" or (party == "PSP"):
            return "opposition"

        # Cabinet members (PAP with ministerial/political appointment)
        cabinet_keywords = [
            "Prime Minister", "Deputy Prime Minister", "Senior Minister",
            "Coordinating Minister", "Minister for", "Minister of State",
            "Senior Minister of State", "Parliamentary Secretary",
            "Senior Parliamentary Secretary", "Acting Minister",
        ]
        if party == "PAP":
            for kw in cabinet_keywords:
                if kw in title:
                    return "cabinet"
            # Speaker/Deputy Speaker are institutional roles
            if "Speaker" in title:
                return "cabinet"
            return "pap_backbench"

        return "pap_backbench"  # fallback

    @staticmethod
    def speaker_type_label(stype: str) -> str:
        """Human-readable label for speaker type."""
        return _SPEAKER_TYPE_LABELS.get(stype, stype)

    def identify_speakers(self, text: str) -> list[dict]:
        """Identify all speakers mentioned in a transcript chunk.

        Returns list of dicts: {name, title, party, constituency, speaker_type, type_label}
        """
        import re
        speakers = []
        seen = set()
        title_pattern = re.compile(
            r"(?:Mr|Mrs|Ms|Madam|Mdm|Dr|Professor|Prof|Associate Professor|Assoc Prof)"
            r"\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})"
        )
        for match in title_pattern.finditer(text):
            name = match.group(1)
            mp = self.find_closest(name)
            if mp and mp["name"] not in seen:
                seen.add(mp["name"])
                stype = self.speaker_type(mp)
                speakers.append({
                    **mp,
                    "speaker_type": stype,
                    "type_label": self.speaker_type_label(stype),
                })
        return speakers


def cleanup_transcript(text: str, mp_index: MPIndex) -> str:
    """Clean up Whisper transcript: fix MP names, domain terms, abbreviations."""
    import re

    # 1. Domain term corrections (longest first to avoid partial matches)
    for wrong, right in sorted(TERM_CORRECTIONS.items(), key=lambda x: -len(x[0])):
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

    # 2. Abbreviation normalisation (longest first)
    for wrong, right in sorted(
        ABBREVIATION_CORRECTIONS.items(), key=lambda x: -len(x[0])
    ):
        text = re.sub(r"\b" + re.escape(wrong) + r"\b", right, text)

    # 3. MP name correction — look for "Mr/Ms/Madam/Dr [Name]" patterns
    title_pattern = re.compile(
        r"(Mr|Mrs|Ms|Madam|Dr|Professor|Prof)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})"
    )
    for match in title_pattern.finditer(text):
        title = match.group(1)
        name = match.group(2)
        corrected = mp_index.find_closest_name(name)
        if corrected and corrected.lower() != name.lower():
            # Only replace the name part, keep the title
            text = text.replace(match.group(0), f"{title} {corrected}")

    return text


# ─── Shared State ─────────────────────────────────────────────────────────────


@dataclass
class ScoredChunk:
    """A transcript chunk with its newsworthiness score."""

    chunk: dict  # original transcript entry
    score: int  # 1-5 nose-for-news-o-meter
    reason: str  # why this score
    line_idx: int  # position in transcript.jsonl
    dispatched: bool = False  # already sent to a blogger


@dataclass
class PipelineState:
    """Shared state between listener, coordinator, and bloggers."""

    transcript_lock: threading.Lock = field(default_factory=threading.Lock)
    blog_lock: threading.Lock = field(default_factory=threading.Lock)
    cursor: int = 0  # next untriaged line index
    total_lines: int = 0  # lines in transcript.jsonl
    new_data: threading.Event = field(default_factory=threading.Event)
    shutdown: threading.Event = field(default_factory=threading.Event)
    last_dispatch_time: float = field(default_factory=time.time)
    pending_futures: list = field(default_factory=list)
    # Nose-for-news-o-meter state
    scored_queue: list = field(default_factory=list)  # list of ScoredChunk
    last_sweep_time: float = field(default_factory=time.time)


# ─── Audio Capture ────────────────────────────────────────────────────────────


def get_audio_stream_url(youtube_url: str) -> str:
    """Extract the best audio stream URL from a YouTube video using yt-dlp."""
    print(f"[listener] Extracting audio stream from: {youtube_url}")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestaudio/worst", "-g", "--no-warnings", youtube_url],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")
    url = result.stdout.strip()
    if not url:
        raise RuntimeError("yt-dlp returned no audio URL")
    print(f"[listener] Audio stream URL obtained")
    return url


def capture_audio_chunk(
    stream_url: str, output_path: str, duration: int = 30
) -> bool:
    """Capture a chunk of audio from the stream using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        stream_url,
        "-t",
        str(duration),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        output_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=duration + 30
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "Server returned 403" in stderr or "HTTP error 403" in stderr:
                return False
            if stderr:
                print(f"  [ffmpeg] Warning: {stderr[:200]}")
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except subprocess.TimeoutExpired:
        print("  [ffmpeg] Chunk capture timed out")
        return False


# ─── Transcription ────────────────────────────────────────────────────────────

DEFAULT_MODEL = "jensenlwt/whisper-small-singlish-122k"


def load_whisper_model(model_name: str):
    """Load a Whisper model via HuggingFace transformers pipeline."""
    print(f"[init] Loading Whisper model: {model_name}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    print(f"[init] Whisper model loaded (device: {device})")
    return pipe


def transcribe_chunk(pipe, audio_path: str) -> str:
    """Transcribe an audio chunk using the HuggingFace ASR pipeline."""
    result = pipe(audio_path)
    return result["text"].strip()


# ─── Speaker Diarisation ─────────────────────────────────────────────────────


def load_diarization_model():
    """Load pyannote speaker diarisation pipeline."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[init] HF_TOKEN not set — speaker diarisation disabled")
        return None
    try:
        from pyannote.audio import Pipeline as PyannotePipeline

        print("[init] Loading pyannote speaker diarisation...")
        diar = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        diar.to(torch.device(device))
        print(f"[init] Diarisation model loaded (device: {device})")
        return diar
    except Exception as e:
        print(f"[init] Diarisation unavailable: {e}")
        return None


def diarize_chunk(diar_pipeline, audio_path: str) -> list[dict]:
    """Run speaker diarisation on an audio chunk.

    Returns list of speaker segments:
    [{"speaker": "SPEAKER_00", "start": 0.5, "end": 12.3}, ...]
    """
    if diar_pipeline is None:
        return []
    try:
        result = diar_pipeline(audio_path)
        # pyannote v4 returns DiarizeOutput; extract the Annotation
        annotation = getattr(result, "speaker_diarization", result)
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
            })
        return segments
    except Exception as e:
        print(f"  [diar] Error: {e}")
        return []


def merge_transcript_with_speakers(
    text: str, segments: list[dict], chunk_duration: float
) -> list[dict]:
    """Merge Whisper transcript with speaker segments.

    Since Whisper gives us a flat string and pyannote gives us speaker time
    ranges, we split the text proportionally across speaker segments.

    Returns list of: [{"speaker": "SPEAKER_00", "text": "..."}, ...]
    """
    if not segments or not text:
        return [{"speaker": "UNKNOWN", "text": text}]

    # Total speaking time
    total_speech = sum(s["end"] - s["start"] for s in segments)
    if total_speech == 0:
        return [{"speaker": "UNKNOWN", "text": text}]

    words = text.split()
    if not words:
        return [{"speaker": segments[0]["speaker"], "text": ""}]

    # Assign words proportionally to speaker segments
    result = []
    word_idx = 0
    for seg in segments:
        seg_duration = seg["end"] - seg["start"]
        proportion = seg_duration / total_speech
        n_words = max(1, round(len(words) * proportion))
        seg_words = words[word_idx : word_idx + n_words]
        word_idx += n_words
        if seg_words:
            result.append({
                "speaker": seg["speaker"],
                "text": " ".join(seg_words),
            })

    # Any remaining words go to the last speaker
    if word_idx < len(words) and result:
        result[-1]["text"] += " " + " ".join(words[word_idx:])

    # Merge consecutive segments from the same speaker
    merged = []
    for seg in result:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append(seg)

    return merged


# ─── Video Frame Speaker Identification ──────────────────────────────────────

FACE_SHOTS_DIR = "face_shots"


def get_video_stream_url(youtube_url: str) -> str | None:
    """Extract a video stream URL from YouTube for frame capture."""
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "93/92/91/worst", "-g", "--no-warnings", youtube_url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def capture_video_frame(stream_url: str, output_path: str) -> bool:
    """Capture a single frame from the video stream."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", stream_url,
        "-vframes", "1",
        "-f", "image2",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 500
    except Exception:
        return False


def load_mp_face_encodings(faces_dir: str = "mp_faces") -> dict:
    """Load MP face photos and create face encodings for matching.

    Returns dict mapping MP name → face encoding (128-dim numpy array).
    """
    import face_recognition

    index_path = os.path.join(faces_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"[face-id] No face index at {index_path}")
        return {}

    with open(index_path) as f:
        index = json.load(f)

    encodings = {}
    loaded = 0
    for filename, name in index.items():
        photo_path = os.path.join(faces_dir, filename)
        if not os.path.exists(photo_path):
            continue
        try:
            image = face_recognition.load_image_file(photo_path)
            face_encs = face_recognition.face_encodings(image)
            if face_encs:
                encodings[name] = face_encs[0]
                loaded += 1
        except Exception:
            continue

    print(f"[face-id] Loaded {loaded}/{len(index)} MP face encodings")
    return encodings


def identify_speaker_from_frame(
    frame_path: str, mp_encodings: dict, threshold: float = 0.5
) -> str | None:
    """Identify the speaker by matching their face against MP photo database.

    Uses face_recognition library for accurate face matching.
    Returns the MP name if confident match found, None otherwise.
    """
    import face_recognition

    try:
        frame_image = face_recognition.load_image_file(frame_path)
        frame_encodings = face_recognition.face_encodings(frame_image)
        if not frame_encodings:
            return None

        # Compare against all MP encodings
        best_name = None
        best_distance = 1.0
        for name, encoding in mp_encodings.items():
            distance = face_recognition.face_distance(
                [encoding], frame_encodings[0]
            )[0]
            if distance < best_distance:
                best_distance = distance
                best_name = name

        if best_distance < threshold and best_name:
            return best_name
    except Exception as e:
        print(f"  [face-id] Error: {e}")
    return None


def save_face_shot(frame_path: str, identified_name: str | None):
    """Save captured frame to face_shots/ with timestamp for verification."""
    os.makedirs(FACE_SHOTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_slug = (identified_name or "unknown").lower().replace(" ", "-")
    dest = os.path.join(FACE_SHOTS_DIR, f"{ts}_{name_slug}.jpg")
    try:
        import shutil
        shutil.copy2(frame_path, dest)
    except Exception:
        pass


# ─── Transcript Storage ──────────────────────────────────────────────────────


def append_transcript(
    jsonl_path: str,
    timestamp: datetime,
    elapsed: timedelta,
    text: str,
    is_silence: bool,
    lock: threading.Lock,
    speakers: list[dict] | None = None,
    identified_speaker: str | None = None,
):
    """Append a transcript chunk to the JSONL file (thread-safe)."""
    entry = {
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "elapsed": str(elapsed).split(".")[0],
        "text": text,
        "is_silence": is_silence,
    }
    if speakers:
        entry["speakers"] = speakers
    if identified_speaker:
        entry["identified_speaker"] = identified_speaker
    with lock:
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    return entry


def read_transcript_lines(jsonl_path: str, lock: threading.Lock) -> list[dict]:
    """Read all transcript lines from the JSONL file (thread-safe)."""
    with lock:
        if not os.path.exists(jsonl_path):
            return []
        lines = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines


# ─── Blog Generation (Gemini) ────────────────────────────────────────────────


def create_gemini_client() -> genai.Client:
    """Create a Gemini API client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[error] GOOGLE_API_KEY environment variable not set")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def read_last_blog_entries(blog_path: str, n: int = 5) -> str:
    """Read the last N entries from the liveblog markdown file."""
    if not os.path.exists(blog_path):
        return "(No entries yet)"
    with open(blog_path) as f:
        content = f.read()
    # Split on double newlines to find entry boundaries
    parts = content.split("\n---\n")
    if len(parts) <= 1:
        return "(No entries yet)"
    entries = [p.strip() for p in parts[1:] if p.strip()]
    if not entries:
        return "(No entries yet)"
    return "\n\n---\n\n".join(entries[-n:])


def call_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str | None:
    """Call Gemini API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )
            if response.text:
                return response.text.strip()
            return None
        except Exception as e:
            wait = 2**attempt
            print(f"  [gemini] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  [gemini] Retrying in {wait}s...")
                time.sleep(wait)
    return None


def triage_chunks(
    client: genai.Client,
    chunks: list[dict],
    start_idx: int,
    mp_index: MPIndex | None = None,
) -> list[ScoredChunk]:
    """Call Gemini to score transcript chunks on the nose-for-news-o-meter.

    Identifies speakers in each chunk and passes their type (cabinet, backbench,
    opposition, NMP) as context so the triage model scores accordingly.

    Returns a list of ScoredChunk with scores 1-5.
    """
    content_texts = []
    chunk_map = []  # maps content index → (original chunk, line index)
    all_speakers = {}  # name → speaker info (deduplicated across chunks)

    for i, chunk in enumerate(chunks):
        if chunk.get("is_silence", False):
            continue
        # Identify speakers in this chunk
        speaker_tag = ""

        # Use video-identified speaker if available
        vid_speaker = chunk.get("identified_speaker")
        if vid_speaker and mp_index:
            match = mp_index.find_closest(vid_speaker)
            if match:
                stype = mp_index.speaker_type(match)
                type_label = mp_index.speaker_type_label(stype)
                enriched = {**match, "speaker_type": stype, "type_label": type_label}
                all_speakers[match["name"]] = enriched
                speaker_tag = f" [Speaker: {match['name']} — {type_label}]"
            else:
                speaker_tag = f" [Speaker: {vid_speaker}]"
        elif vid_speaker:
            speaker_tag = f" [Speaker: {vid_speaker}]"

        # Also check for names mentioned in text
        if mp_index:
            mentioned = mp_index.identify_speakers(chunk.get("text", ""))
            for sp in mentioned:
                all_speakers[sp["name"]] = sp
                if not speaker_tag:
                    speaker_tag = f" [Speaker: {sp['name']} — {sp['type_label']}]"

        content_texts.append(
            f"[Chunk {len(content_texts)}] [{chunk['timestamp']}]{speaker_tag} {chunk['text']}"
        )
        chunk_map.append((chunk, start_idx + i))

    if not content_texts:
        # All silence — score them as 1
        return [
            ScoredChunk(chunk=c, score=1, reason="silence", line_idx=start_idx + i)
            for i, c in enumerate(chunks)
        ]

    # Build speaker context summary for the triage prompt
    speaker_context = ""
    if all_speakers:
        speaker_lines = []
        for sp in all_speakers.values():
            speaker_lines.append(
                f"- {sp['honorific']} {sp['name']}: {sp['title']}, "
                f"{sp['constituency']}, {sp['party']} → {sp['type_label']}"
            )
        speaker_context = (
            "## Speakers identified in these chunks:\n" + "\n".join(speaker_lines)
        )

    chunks_text = "\n".join(content_texts)
    raw = call_gemini(
        client,
        TRIAGE_SYSTEM_PROMPT,
        TRIAGE_USER_PROMPT.format(speaker_context=speaker_context, chunks=chunks_text),
    )

    scored = []
    if raw:
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            data = json.loads(cleaned.strip())
            assessments = data.get("assessments", [])
            # Build lookup by chunk_idx
            score_map = {}
            for a in assessments:
                idx = a.get("chunk_idx", -1)
                score_map[idx] = (
                    min(max(a.get("score", 1), 1), 5),
                    a.get("reason", ""),
                )
            # Map back to original chunks
            for content_idx, (chunk, line_idx) in enumerate(chunk_map):
                score, reason = score_map.get(content_idx, (2, "unscored"))
                scored.append(
                    ScoredChunk(
                        chunk=chunk, score=score, reason=reason, line_idx=line_idx
                    )
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  [triage] Failed to parse scores: {e}")
            # Fallback: score everything as 3 (notable) so nothing is lost
            for content_idx, (chunk, line_idx) in enumerate(chunk_map):
                scored.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=3,
                        reason="triage parse failed — defaulting to notable",
                        line_idx=line_idx,
                    )
                )
    else:
        # Gemini call failed — score everything as 3
        for content_idx, (chunk, line_idx) in enumerate(chunk_map):
            scored.append(
                ScoredChunk(
                    chunk=chunk,
                    score=3,
                    reason="triage call failed — defaulting to notable",
                    line_idx=line_idx,
                )
            )

    # Print the scores
    for sc in scored:
        stars = "*" * sc.score
        label = {5: "BREAKING", 4: "STRONG", 3: "NOTABLE", 2: "COLOUR", 1: "ROUTINE"}
        preview = sc.chunk.get("text", "")[:80]
        print(
            f"  [triage] {stars:<5} {label.get(sc.score, '?'):<8} {preview}{'...' if len(sc.chunk.get('text','')) > 80 else ''}"
        )
        if sc.score >= 3:
            print(f"           → {sc.reason}")

    return scored


def generate_blog_entries(
    client: genai.Client,
    chunks: list[dict],
    blog_path: str,
    score_context: str = "",
) -> str | None:
    """Call Gemini to generate live blog entries from transcript chunks."""
    existing = read_last_blog_entries(blog_path)
    chunk_lines = []
    for c in chunks:
        speaker = c.get("identified_speaker", "")
        speaker_tag = f" [Speaker: {speaker}]" if speaker else ""
        chunk_lines.append(
            f"[{c['timestamp']}] (elapsed: {c['elapsed']}){speaker_tag} {c['text']}"
        )
    chunks_text = "\n".join(chunk_lines)

    text = call_gemini(
        client,
        LIVE_BLOG_SYSTEM_PROMPT,
        LIVE_BLOG_USER_PROMPT.format(
            existing_entries=existing,
            chunks=chunks_text,
            score_context=score_context,
        ),
    )

    if not text or text.strip() == "NO_UPDATE":
        return None
    return text


def run_editor_pass(client: genai.Client, entry: str) -> str:
    """Run the editor pass to add <aside> context blocks to a blog entry."""
    result = call_gemini(
        client,
        EDITOR_SYSTEM_PROMPT,
        EDITOR_USER_PROMPT.format(entry=entry),
    )
    if result:
        return result
    return entry  # Fallback to original if editor pass fails


# ─── Blog File Management ────────────────────────────────────────────────────


def init_blog_file(blog_path: str):
    """Initialise the liveblog markdown file with a header."""
    if os.path.exists(blog_path):
        return
    today = datetime.now().strftime("%b %-d, %Y")
    header = f"""# Live Blog: Parliament Sitting — {today}

*Live coverage of Singapore Parliament proceedings*

---

"""
    with open(blog_path, "w") as f:
        f.write(header)


def append_blog_entries(blog_path: str, entries: str, lock: threading.Lock):
    """Append new blog entries to the liveblog file (thread-safe)."""
    now = datetime.now().strftime("%-I:%M%p").lower()
    with lock:
        with open(blog_path, "a") as f:
            f.write(f"\n**[{now}]**\n\n{entries}\n\n---\n")


# ─── Dispatch Logic (News-Score-Driven) ───────────────────────────────────────


def check_dispatch(state: PipelineState) -> list[tuple[list[ScoredChunk], str]]:
    """Check scored queue and decide what to dispatch to bloggers.

    Returns a list of (batch, reason) tuples. Multiple dispatches possible
    if e.g. there's a score-5 AND accumulated score-3s.

    Dispatch rules:
    - Score 5 (BREAKING): dispatch immediately, alone or with recent context
    - Score 4 (STRONG):   dispatch immediately with nearby chunks for context
    - Score 3 (NOTABLE):  dispatch when 2+ accumulate, or on silence break,
                          or after 3 min with no dispatch
    - Score 2 (COLOUR):   swept into nearby batches as background, never alone
    - Score 1 (ROUTINE):  never dispatched
    """
    dispatches = []
    undispatched = [sc for sc in state.scored_queue if not sc.dispatched]
    if not undispatched:
        return dispatches

    # Separate by score tier
    breaking = [sc for sc in undispatched if sc.score == 5]
    strong = [sc for sc in undispatched if sc.score == 4]
    notable = [sc for sc in undispatched if sc.score == 3]
    colour = [sc for sc in undispatched if sc.score == 2]

    def gather_context(anchor: ScoredChunk, radius: int = 2) -> list[ScoredChunk]:
        """Gather nearby undispatched chunks (score >= 2) around an anchor."""
        batch = []
        for sc in undispatched:
            if sc.dispatched:
                continue
            if abs(sc.line_idx - anchor.line_idx) <= radius and sc.score >= 2:
                batch.append(sc)
        if anchor not in batch:
            batch.append(anchor)
        batch.sort(key=lambda x: x.line_idx)
        return batch

    # 1. BREAKING (5) — dispatch immediately with context
    for sc in breaking:
        if sc.dispatched:
            continue
        batch = gather_context(sc, radius=3)
        for b in batch:
            b.dispatched = True
        dispatches.append((batch, f"BREAKING [{sc.score}]: {sc.reason}"))

    # 2. STRONG (4) — dispatch immediately with context
    for sc in strong:
        if sc.dispatched:
            continue
        batch = gather_context(sc, radius=2)
        for b in batch:
            b.dispatched = True
        dispatches.append((batch, f"STRONG NEWS [{sc.score}]: {sc.reason}"))

    # 3. NOTABLE (3) — batch when 2+ accumulate, or time/silence triggers
    remaining_notable = [sc for sc in notable if not sc.dispatched]
    if len(remaining_notable) >= 2:
        # Batch all notables together, pull in adjacent colour
        batch = list(remaining_notable)
        for c in colour:
            if not c.dispatched:
                # Include colour if adjacent to any notable
                for n in remaining_notable:
                    if abs(c.line_idx - n.line_idx) <= 2:
                        batch.append(c)
                        break
        batch = list({id(sc): sc for sc in batch}.values())  # dedupe
        batch.sort(key=lambda x: x.line_idx)
        for b in batch:
            b.dispatched = True
        reasons = [sc.reason for sc in remaining_notable[:3]]
        dispatches.append(
            (batch, f"NOTABLE batch ({len(remaining_notable)} items): {'; '.join(reasons)}")
        )
    elif remaining_notable:
        # Only 1 notable — check time-based triggers
        elapsed = time.time() - state.last_dispatch_time
        if elapsed >= 180:
            batch = list(remaining_notable)
            for c in colour:
                if not c.dispatched and any(
                    abs(c.line_idx - n.line_idx) <= 2 for n in remaining_notable
                ):
                    batch.append(c)
            batch = list({id(sc): sc for sc in batch}.values())
            batch.sort(key=lambda x: x.line_idx)
            for b in batch:
                b.dispatched = True
            dispatches.append(
                (batch, f"NOTABLE (time sweep, {elapsed:.0f}s): {remaining_notable[0].reason}")
            )

    # 4. Periodic sweep — catch stale colour (2) that's been sitting > 5 min
    remaining_colour = [sc for sc in colour if not sc.dispatched]
    sweep_elapsed = time.time() - state.last_sweep_time
    if remaining_colour and sweep_elapsed >= 300:
        # Only if there's genuinely interesting colour (multiple items)
        if len(remaining_colour) >= 3:
            batch = remaining_colour
            batch.sort(key=lambda x: x.line_idx)
            for b in batch:
                b.dispatched = True
            state.last_sweep_time = time.time()
            dispatches.append(
                (batch, f"COLOUR sweep ({len(remaining_colour)} items, {sweep_elapsed:.0f}s)")
            )

    return dispatches


# ─── Listener Thread ─────────────────────────────────────────────────────────


def listener_thread(
    youtube_url: str,
    chunk_seconds: int,
    whisper_model,
    diar_model,
    mp_encodings: dict,
    transcript_path: str,
    state: PipelineState,
):
    """Listener thread: captures audio, transcribes, diarises, identifies speaker, appends JSONL.

    This thread NEVER blocks on blog generation. It runs continuously until
    shutdown is requested.
    """
    chunk_count = 0
    start_time = datetime.now()
    stream_url = None
    stream_url_time = None
    video_stream_url = None
    stream_refresh_interval = 3600
    tmpdir = tempfile.mkdtemp(prefix="parliament_")
    last_identified_speaker = None  # cache last identified speaker
    consecutive_failures = 0

    try:
        while not state.shutdown.is_set():
            # Get or refresh stream URL
            if stream_url is None or (
                time.time() - stream_url_time > stream_refresh_interval
            ):
                try:
                    stream_url = get_audio_stream_url(youtube_url)
                    video_stream_url = get_video_stream_url(youtube_url)
                    consecutive_failures = 0  # reset on success
                    stream_url_time = time.time()
                    if video_stream_url:
                        print("[listener] Video stream URL obtained (speaker ID enabled)")
                except Exception as e:
                    error_msg = str(e)
                    # Detect stream ended vs transient error
                    stream_ended_signals = [
                        "is not a valid URL",
                        "Video unavailable",
                        "This live event has ended",
                        "live has ended",
                        "is offline",
                        "Private video",
                    ]
                    if any(sig in error_msg for sig in stream_ended_signals):
                        print(f"\n[listener] Live stream has ended: {error_msg}")
                        print("[listener] Signalling shutdown...")
                        state.shutdown.set()
                        break
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print(f"\n[listener] {consecutive_failures} consecutive failures — stream likely ended")
                        print("[listener] Signalling shutdown...")
                        state.shutdown.set()
                        break
                    print(f"[listener] Error getting stream URL: {error_msg}")
                    print(f"[listener] Retrying in 10s... (attempt {consecutive_failures}/5)")
                    state.shutdown.wait(10)
                    continue

            # Capture audio chunk
            chunk_path = os.path.join(tmpdir, f"chunk_{chunk_count:06d}.wav")
            elapsed = datetime.now() - start_time
            now = datetime.now()

            print(
                f"[listener] Recording chunk {chunk_count} "
                f"(elapsed: {str(elapsed).split('.')[0]})...",
                end=" ",
                flush=True,
            )

            # Start audio capture as background process so we can grab
            # a video frame 5 seconds in (while the speaker is on screen)
            audio_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", stream_url,
                "-t", str(chunk_seconds),
                "-vn", "-ac", "1", "-ar", "16000",
                "-acodec", "pcm_s16le",
                chunk_path,
            ]
            try:
                audio_proc = subprocess.Popen(
                    audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except Exception as e:
                print(f"failed to start ffmpeg: {e}")
                stream_url = None
                continue

            # Wait 5 seconds into the chunk, then capture video frame
            identified_speaker = last_identified_speaker
            frame_path = os.path.join(tmpdir, "frame.jpg")
            if video_stream_url and mp_encodings:
                time.sleep(5)
                if capture_video_frame(video_stream_url, frame_path):
                    name = identify_speaker_from_frame(frame_path, mp_encodings)
                    speaker_changed = (name != last_identified_speaker)
                    if name:
                        if speaker_changed:
                            print(f"  [face-id] Speaker: {name}")
                        identified_speaker = name
                        last_identified_speaker = name
                    else:
                        print("  [face-id] No confident match")
                    # Only save face shot when speaker changes (or unknown)
                    if speaker_changed:
                        save_face_shot(frame_path, name)
                    try:
                        os.remove(frame_path)
                    except OSError:
                        pass
                else:
                    print("  [face-id] Frame capture failed")

            # Wait for audio capture to finish
            try:
                _, stderr = audio_proc.communicate(timeout=chunk_seconds + 30)
                returncode = audio_proc.returncode
            except subprocess.TimeoutExpired:
                audio_proc.kill()
                audio_proc.communicate()
                print("  [ffmpeg] Chunk capture timed out")
                continue

            if returncode != 0:
                stderr_text = stderr.decode(errors="replace").strip() if stderr else ""
                if "Server returned 403" in stderr_text or "HTTP error 403" in stderr_text:
                    print("failed (403)")
                    print("[listener] Refreshing stream URL...")
                    stream_url = None
                    continue
                if stderr_text:
                    print(f"  [ffmpeg] Warning: {stderr_text[:200]}")

            success = os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000
            if not success:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(f"failed ({consecutive_failures} consecutive — stream likely ended)")
                    state.shutdown.set()
                    break
                print("failed")
                print(f"[listener] Refreshing stream URL... (attempt {consecutive_failures}/5)")
                stream_url = None
                continue

            consecutive_failures = 0  # successful capture
            # Transcribe
            text = transcribe_chunk(whisper_model, chunk_path)

            # Diarise (runs on same audio file)
            speaker_segments = diarize_chunk(diar_model, chunk_path)

            # Clean up audio file
            try:
                os.remove(chunk_path)
            except OSError:
                pass

            if not text or text.isspace():
                print("(silence)")
                append_transcript(
                    transcript_path, now, elapsed, "", True, state.transcript_lock
                )
                state.total_lines += 1
                state.new_data.set()
                chunk_count += 1
                continue

            # Merge transcript with speaker segments
            speakers = merge_transcript_with_speakers(
                text, speaker_segments, chunk_seconds
            )
            if len(speakers) > 1:
                speaker_summary = " | ".join(
                    f"{s['speaker']}: {s['text'][:40]}..." for s in speakers
                )
                print(f"diar:[{len(speakers)} speakers]")
                print(f"  → {speaker_summary[:120]}")
            else:
                print("ok")

            # Store transcript with speaker segments and identified speaker
            append_transcript(
                transcript_path, now, elapsed, text, False, state.transcript_lock,
                speakers=speakers if len(speakers) > 1 else None,
                identified_speaker=identified_speaker,
            )
            state.total_lines += 1
            chunk_count += 1
            print(f"  → {text[:120]}{'...' if len(text) > 120 else ''}")

            # Signal coordinator that new data is available
            state.new_data.set()

    except Exception as e:
        print(f"\n[listener] Fatal error: {e}")
        state.shutdown.set()
    finally:
        # Clean up temp directory
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
        print("[listener] Thread stopped")


# ─── Blogger Task ─────────────────────────────────────────────────────────────


def blogger_task(
    client: genai.Client,
    chunks: list[dict],
    blog_path: str,
    blog_lock: threading.Lock,
    task_id: int,
    score_context: str = "",
):
    """A single blogger task: generate draft → editor pass → append to blog.

    Runs in the ThreadPoolExecutor.
    """
    content_chunks = [c for c in chunks if not c.get("is_silence", False)]
    if not content_chunks:
        return

    print(f"  [blogger-{task_id}] Generating draft from {len(content_chunks)} chunks...")

    # Step 1: Generate draft (with score context so blogger knows priority)
    draft = generate_blog_entries(client, content_chunks, blog_path, score_context)
    if not draft:
        print(f"  [blogger-{task_id}] No newsworthy updates")
        return

    print(f"  [blogger-{task_id}] Draft ready, running editor pass...")

    # Step 2: Editor pass — add <aside> context blocks
    edited = run_editor_pass(client, draft)

    # Step 3: Append to liveblog.md (thread-safe)
    append_blog_entries(blog_path, edited, blog_lock)
    preview = edited[:200].replace("\n", " ")
    print(f"  [blogger-{task_id}] Entry published")
    print(f"    → {preview}{'...' if len(edited) > 200 else ''}")


# ─── Coordinator (Main Thread) ───────────────────────────────────────────────


def run_coordinator(
    state: PipelineState,
    transcript_path: str,
    blog_path: str,
    gemini_client: genai.Client,
    max_bloggers: int,
    mp_index: MPIndex | None = None,
):
    """Coordinator: triages new chunks, dispatches bloggers based on news scores.

    Flow:
    1. Wait for new transcript data from listener
    2. Triage new chunks via Gemini (cheap call → nose-for-news-o-meter scores)
    3. Check dispatch rules based on scores
    4. Submit batches to blogger pool
    """
    task_counter = 0
    corrections = CorrectionCache()

    with ThreadPoolExecutor(max_workers=max_bloggers, thread_name_prefix="blogger") as pool:
        while not state.shutdown.is_set():
            # Wait for new data (or periodic check for time-based dispatch)
            signalled = state.new_data.wait(timeout=5.0)
            if signalled:
                state.new_data.clear()

            # Read current transcript
            lines = read_transcript_lines(transcript_path, state.transcript_lock)
            if not lines:
                continue

            # Triage any new untriaged chunks
            if state.cursor < len(lines):
                new_chunks = lines[state.cursor :]
                start_idx = state.cursor
                state.cursor = len(lines)

                # Apply corrections.json to chunk text before triage/blogging
                for chunk in new_chunks:
                    if chunk.get("text"):
                        chunk["text"] = corrections.apply(chunk["text"])

                print(f"\n[coordinator] Triaging {len(new_chunks)} new chunk(s)...")
                # Split large batches to avoid Gemini truncating JSON output
                TRIAGE_BATCH_SIZE = 8
                if len(new_chunks) > TRIAGE_BATCH_SIZE:
                    all_scored = []
                    for batch_start in range(0, len(new_chunks), TRIAGE_BATCH_SIZE):
                        batch = new_chunks[batch_start:batch_start + TRIAGE_BATCH_SIZE]
                        batch_idx = start_idx + batch_start
                        scored = triage_chunks(gemini_client, batch, batch_idx, mp_index)
                        all_scored.extend(scored)
                    state.scored_queue.extend(all_scored)
                else:
                    scored = triage_chunks(gemini_client, new_chunks, start_idx, mp_index)
                    state.scored_queue.extend(scored)

            # Check dispatch rules (also runs on timeout for time-based sweeps)
            dispatches = check_dispatch(state)

            for batch, reason in dispatches:
                task_counter += 1
                content_count = sum(1 for sc in batch if sc.score >= 2)
                print(f"\n[coordinator] Dispatching blogger-{task_counter}: {reason}")
                print(f"  ({content_count} content chunks, scores: {[sc.score for sc in batch]})")

                # Build score context for the blogger prompt
                score_lines = []
                for sc in batch:
                    label = {5: "BREAKING", 4: "STRONG", 3: "NOTABLE", 2: "COLOUR", 1: "ROUTINE"}
                    score_lines.append(
                        f"- [{label.get(sc.score, '?')} {sc.score}/5] {sc.reason}"
                    )
                score_ctx = (
                    "## Editorial triage (nose-for-news-o-meter scores for these chunks):\n"
                    + "\n".join(score_lines)
                    + "\n\nPrioritise BREAKING and STRONG items. Include NOTABLE items."
                    " COLOUR items can be woven in for texture but don't need their own entry."
                )

                raw_chunks = [sc.chunk for sc in batch]
                state.last_dispatch_time = time.time()

                future = pool.submit(
                    blogger_task,
                    gemini_client,
                    raw_chunks,
                    blog_path,
                    state.blog_lock,
                    task_counter,
                    score_ctx,
                )
                state.pending_futures.append(future)

            # Clean up completed futures
            state.pending_futures = [f for f in state.pending_futures if not f.done()]

        # Shutdown: dispatch any remaining scored-but-undispatched chunks (score >= 3)
        print("\n[coordinator] Processing remaining chunks before shutdown...")
        remaining = [
            sc for sc in state.scored_queue if not sc.dispatched and sc.score >= 2
        ]
        if remaining:
            task_counter += 1
            scores = [sc.score for sc in remaining]
            print(f"[coordinator] Final batch: {len(remaining)} chunks (scores: {scores})")
            for sc in remaining:
                sc.dispatched = True
            raw_chunks = [sc.chunk for sc in remaining]
            future = pool.submit(
                blogger_task,
                gemini_client,
                raw_chunks,
                blog_path,
                state.blog_lock,
                task_counter,
            )
            state.pending_futures.append(future)

        # Also triage+dispatch any chunks that arrived after the last triage
        lines = read_transcript_lines(transcript_path, state.transcript_lock)
        if state.cursor < len(lines):
            new_chunks = lines[state.cursor :]
            content = [c for c in new_chunks if not c.get("is_silence", False)]
            if content:
                task_counter += 1
                print(f"[coordinator] Final untriaged: {len(content)} content chunks")
                future = pool.submit(
                    blogger_task,
                    gemini_client,
                    content,
                    blog_path,
                    state.blog_lock,
                    task_counter,
                )
                state.pending_futures.append(future)

        # Wait for all pending bloggers to finish
        for future in state.pending_futures:
            try:
                future.result(timeout=120)
            except Exception as e:
                print(f"[coordinator] Blogger error: {e}")

        print("[coordinator] All bloggers finished")


# ─── Main Pipeline ────────────────────────────────────────────────────────────


def run_pipeline(args):
    youtube_url = args.url
    model_name = args.model
    chunk_seconds = args.chunk_seconds
    max_bloggers = args.max_bloggers
    transcript_path = args.transcript or "transcript.jsonl"
    blog_path = args.blog or "liveblog.md"
    mp_list_path = args.mp_list or "mp_list.json"

    # Initialise
    whisper_model = load_whisper_model(model_name)
    diar_model = load_diarization_model()
    mp_encodings = load_mp_face_encodings()
    gemini_client = create_gemini_client()
    mp_index = MPIndex(mp_list_path)
    init_blog_file(blog_path)

    # Shared state
    state = PipelineState()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[pipeline] Shutdown requested. Finishing current work...")
        state.shutdown.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[pipeline] Starting live pipeline (threaded)")
    print(f"[pipeline] Transcript: {transcript_path}")
    print(f"[pipeline] Live blog:  {blog_path}")
    print(f"[pipeline] MP list:    {mp_list_path} ({len(mp_index.mps)} MPs loaded)")
    print(f"[pipeline] Diarisation: {'ON' if diar_model else 'OFF'}")
    print(f"[pipeline] Face ID:    ON ({len(mp_encodings)} MPs loaded)" if mp_encodings else "[pipeline] Face ID:    OFF (no encodings)")
    print(f"[pipeline] Chunk size: {chunk_seconds}s")
    print(f"[pipeline] Max bloggers: {max_bloggers}")
    print(f"[pipeline] Press Ctrl+C to stop\n")

    # Start listener thread (daemon so it dies with main)
    listener = threading.Thread(
        target=listener_thread,
        args=(
            youtube_url,
            chunk_seconds,
            whisper_model,
            diar_model,
            mp_encodings,
            transcript_path,
            state,
        ),
        name="listener",
        daemon=True,
    )
    listener.start()

    # Run coordinator on main thread
    try:
        run_coordinator(state, transcript_path, blog_path, gemini_client, max_bloggers, mp_index)
    except KeyboardInterrupt:
        state.shutdown.set()

    print(f"\n[pipeline] Done. Transcript: {transcript_path}")
    print(f"[pipeline] Live blog: {blog_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Live pipeline: YouTube → transcription → auto live blog (threaded)"
    )
    parser.add_argument("url", help="YouTube video or livestream URL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace Whisper model ID. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=30,
        help="Audio chunk duration in seconds. Default: 30",
    )
    parser.add_argument(
        "--max-bloggers",
        type=int,
        default=3,
        help="Maximum concurrent blogger threads. Default: 3",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="Path to transcript JSONL file. Default: transcript.jsonl",
    )
    parser.add_argument(
        "--blog",
        default=None,
        help="Path to liveblog markdown file. Default: liveblog.md",
    )
    parser.add_argument(
        "--mp-list",
        default=None,
        help="Path to MP list JSON file. Default: mp_list.json",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
