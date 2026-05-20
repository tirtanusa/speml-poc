from flask import Flask, render_template, request, jsonify, send_file
from nltk import edit_distance
from dotenv import load_dotenv
import requests as http_requests
import json
import re
import os

# ===== Load environment variables dari .env =====
load_dotenv()

TRANSLATE_URL   = os.getenv("TRANSLATE_URL", "")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
print("TRANSLATE_URL", TRANSLATE_URL)
print("NGROK_AUTH_TOKEN", NGROK_AUTH_TOKEN)

app = Flask(__name__)

# ===== Load vocab for Levenshtein & Jaro-Winkler =====
with open("vocabulary.json", "r", encoding="utf-8") as f:
    VOCAB = set(json.load(f))

# =====================================================
# ============ Algoritma Peter Norvig =================
# =====================================================

def words(text):
    return re.findall(r'\w+', text.lower())

def load_vocabulary(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # Asumsi format JSON: {"word": count, ...}
    word_counts = {}
    for word, count in data.items():
        word_counts[word.lower()] = count
    return word_counts

# Load vocabulary Norvig
WORDS = load_vocabulary("vocabulary.json")

def P(word, N=sum(WORDS.values())):
    return WORDS.get(word, 0) / N

def correction(word):
    return max(candidates(word), key=P)

def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# =====================================================
# ============ Algoritma Levenshtein ==================
# =====================================================

def lev_distance(kata, vocab):
    kata_terdekat = kata
    min_distance = float("inf")

    for vocab_kata in vocab:
        distance = edit_distance(kata, vocab_kata)
        if distance < min_distance:
            min_distance = distance
            kata_terdekat = vocab_kata

    return kata_terdekat


# =====================================================
# ============ Algoritma Jaro-Winkler =================
# =====================================================

def jaro_winkler(s1, s2):
    if not isinstance(s1, str) or not isinstance(s2, str):
        return 0.0

    s1, s2 = s1.lower(), s2.lower()
    len1, len2 = len(s1), len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    match_window = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i, ch1 in enumerate(s1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)

        for j in range(start, end):
            if not s2_matches[j] and s2[j] == ch1:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    s1_match_chars = [s1[i] for i in range(len1) if s1_matches[i]]
    s2_match_chars = [s2[i] for i in range(len2) if s2_matches[i]]

    for c1, c2 in zip(s1_match_chars, s2_match_chars):
        if c1 != c2:
            transpositions += 1

    transpositions //= 2

    jaro = ((matches / len1) + (matches / len2) + ((matches - transpositions) / matches)) / 3

    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix += 1
        else:
            break
    prefix = min(4, prefix)

    p = 0.1
    return jaro + prefix * p * (1 - jaro)


def kata_terbaik_jaro(kata, vocab):
    kata_terdekat = kata
    max_score = 0.0

    for vocab_kata in vocab:
        score = jaro_winkler(kata, vocab_kata)
        if score > max_score:
            max_score = score
            kata_terdekat = vocab_kata

    return kata_terdekat


# =====================================================
# ================== Spell Checker ====================
# =====================================================

def spellcheck(kalimat, vocab, algorithm="levenshtein"):
    if not isinstance(kalimat, str) or not kalimat:
        return "Input harus berupa string dan tidak boleh kosong!"

    tokens = re.findall(r'\b\w+\b|[^\w\s]', kalimat)
    hasil = []

    for token in tokens:
        if re.match(r'^\w+$', token):  # token kata
            lower = token.lower()

            if lower in vocab or lower in WORDS:
                hasil.append(token)
            else:
                if algorithm == "levenshtein":
                    corrected = lev_distance(lower, vocab)

                elif algorithm == "jaro-winkler":
                    corrected = kata_terbaik_jaro(lower, vocab)

                elif algorithm == "norvig":
                    corrected = correction(lower)

                else:
                    corrected = token

                if token[0].isupper():
                    corrected = corrected.capitalize()

                hasil.append(corrected)
        else:
            hasil.append(token)

    return " ".join(hasil)


# =====================================================
# ===================== ROUTES ========================
# =====================================================

@app.route("/")
@app.route("/dashboard")
@app.route("/spellchecker")
@app.route("/translation")
@app.route("/combo")
def index():
    return send_file("index.html")


@app.route("/spellcheck", methods=["POST"])
def spellcheck_route():
    data = request.get_json()
    algorithm = data.get("algorithm", "levenshtein")  # levenshtein | jaro-winkler | norvig
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    corrected = spellcheck(text, VOCAB, algorithm)
    return jsonify({"result": corrected})


@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json()
    text = data.get("text", "")
    config = data.get("config", "higher")

    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400

    if not TRANSLATE_URL:
        return jsonify({"error": "TRANSLATE_URL belum dikonfigurasi di file .env"}), 500

    # Bersihkan base URL (hapus trailing slash / terjemahkan jika ada)
    base_url = TRANSLATE_URL.rstrip("/")
    if base_url.endswith("/terjemahkan"):
        base_url = base_url[:-12]

    # Susun URL secara dinamis berdasarkan pilihan LoRA
    if config == "lower":
        target_url = f"{base_url}/model2/terjemahkan"
    else:
        target_url = f"{base_url}/model1/terjemahkan"

    try:
        # Siapkan headers — tambahkan auth token jika tersedia
        headers = {"Content-Type": "application/json"}
        if NGROK_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {NGROK_AUTH_TOKEN}"
            headers["ngrok-skip-browser-warning"] = "true"

        # Forward ke endpoint terjemahan eksternal
        ext_response = http_requests.post(
            target_url,
            json={"teks": text},
            headers=headers,
            timeout=30
        )
        ext_response.raise_for_status()

        ext_data = ext_response.json()

        # Response format: {"hasil": "...", "input": "..."}
        hasil = ext_data.get("hasil", "")
        if not hasil:
            return jsonify({"error": "Response dari server terjemahan tidak mengandung key 'hasil'"}), 502

        return jsonify({"result": hasil})

    except http_requests.exceptions.Timeout:
        return jsonify({"error": "Request ke server terjemahan timeout (>30 detik)"}), 504
    except http_requests.exceptions.ConnectionError:
        return jsonify({"error": "Tidak dapat terhubung ke server terjemahan. Pastikan ngrok aktif."}), 503
    except http_requests.exceptions.HTTPError as e:
        return jsonify({"error": f"Server terjemahan mengembalikan error: {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": f"Error tidak terduga: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
