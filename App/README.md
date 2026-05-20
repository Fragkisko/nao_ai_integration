# NAO Integration — Environment & Run Instructions

Αυτό το αρχείο περιγράφει τις μεταβλητές περιβάλλοντος που χρησιμοποιεί το project και παραδείγματα εκτέλεσης.

## Σημαντικές μεταβλητές περιβάλλοντος

- `NAO_IP` : IP διεύθυνση του NAO (π.χ. `192.168.1.42`). Προεπιλογή: `127.0.0.1`.
- `NAO_PORT` : Port για σύνδεση ALProxy (π.χ. `9559`). Προεπιλογή: `9559`.
- `PY2_EXE` : Διαδρομή στο εκτελέσιμο Python2 που θα τρέξει το helper `py2/nao_say.py`. Προεπιλογή: `C:\Python27\python.exe`.
- `PYNAOQI_LIB` : Φάκελος με τη βιβλιοθήκη `pynaoqi` για Python2. Προεπιλογή: `C:\NaoQi\pynaoqi-...\lib`.
- `NAO_TTS_TIMEOUT_SEC` : Timeout για subprocess TTS σε δευτερόλεπτα. Προεπιλογή: `30`.

- OLLAMA / RAG related:
  - `OLLAMA_URL` : URL του Ollama generate API. Προεπιλογή: `http://localhost:11434/api/generate`.
  - `OLLAMA_MODEL` : Προεπιλογή: `llama3:latest`.
  - `MAX_TTS_LEN` : Μέγιστο μήκος κειμένου για TTS (χαρακτήρες). Προεπιλογή: `250`.
  - `RAG_MODEL` : RAG model override. Προεπιλογή: `gpt-4o-mini`.
  - `RAG_TOP_K` : Πόσα αποτελέσματα RAG να χρησιμοποιηθούν. Προεπιλογή: `5`.
  - `RAG_DEBUG_CHUNKS` : 0/1 or true/false για εκτύπωση chunks.

- Optional tuning:
  - `OLLAMA_SYSTEM`, `OLLAMA_TIMEOUT_SEC`, `OLLAMA_TEMPERATURE` — για παραμετροποίηση Ollama calls.

## Παραδείγματα (προσωρινά μόνο για το session)

PowerShell (προσωρινά για το τρέχον session):

```powershell
$env:NAO_IP="192.168.1.42"
$env:PY2_EXE="C:\Python27\python.exe"
$env:PYNAOQI_LIB="C:\path\to\pynaoqi\lib"
python .\App\main.py
```

CMD (προσωρινά μόνο για την τρέχουσα κονσόλα):

```cmd
set NAO_IP=192.168.1.42
set PY2_EXE=C:\Python27\python.exe
set PYNAOQI_LIB=C:\path\to\pynaoqi\lib
python App\main.py
```

Μόνιμη ρύθμιση (Windows):

```cmd
setx NAO_IP "192.168.1.42"
setx PY2_EXE "C:\Python27\python.exe"
setx PYNAOQI_LIB "C:\path\to\pynaoqi\lib"
```

(Άνοιξε νέο τερματικό μετά το `setx`.)

- Το `nao_common.py` διαβάζει `PY2_EXE`, `PYNAOQI_LIB`, `NAO_IP`, κ.λπ. μέσω `os.environ.get(..., default)` — αν δεν έχεις ορίσει μία μεταβλητή, χρησιμοποιείται η προεπιλογή που υπάρχει στον κώδικα.
- Η `nao_say()` περνάει αυτές τις ρυθμίσεις στο subprocess που τρέχει το helper `py2/nao_say.py`.
