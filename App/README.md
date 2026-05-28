# NAO Integration — Environment & Run Instructions

Αυτό το αρχείο περιγράφει τις μεταβλητές περιβάλλοντος που χρησιμοποιεί το project και παραδείγματα εκτέλεσης.

## Modular δομή (χωρίς αλλαγή στο run command)

Το project χωρίστηκε σε μικρότερα αρχεία για ευκολότερη συντήρηση, αλλά το entrypoint παραμένει ίδιο:

```powershell
python .\App\main.py
```

Νέα αρχεία στον φάκελο `App`:

- `main.py`: orchestrator του app loop (menu, input routing, reply flow).
- `settings.py`: φόρτωση `.env`, CLI overrides (`--nao-ip`, `--nao-port`) και συγκέντρωση ρυθμίσεων.
- `speech_to_text.py`: VOSK loading, push-to-talk, transcoding και file transcription.
- `nao_service.py`: remote NAO recording flow (ssh/scp) + μεταγραφή.
- `rag_service.py`: φόρτωση `rag_cli.py`, init του RAG engine και query execution.

Με αυτό το split, κρατάμε ίδια λειτουργικότητα αλλά έχουμε καθαρό separation of concerns.

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
  - `RAG_PRELOAD_RETRIEVAL_ON_BOOT` : Αν είναι `1/true`, φορτώνει embedding retrieval resources στο startup για να μην υπάρχει καθυστέρηση στο πρώτο document query. Προεπιλογή: `1`.
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

## Φωνητική Είσοδος (Offline STT με VOSK)

Το project υποστηρίζει offline speech-to-text με VOSK και `sounddevice` (push-to-talk). Η λειτουργία αυτή είναι προαιρετική — αν δεν εγκατασταθούν οι βιβλιοθήκες ή το μοντέλο, το πρόγραμμα θα συνεχίσει να λειτουργεί με πληκτρολόγιο μόνο.

Βασικά βήματα για να ενεργοποιήσεις τη φωνητική είσοδο:

- Κατέβασε ένα Greek VOSK model (π.χ. `vosk-model-el-gr-0.7`) από: https://alphacephei.com/vosk/models
- Τοποθέτησε τον αποσυμπιεσμένο φάκελο μοντέλου κάπου στον υπολογιστή σου, π.χ. `C:\Users\frag2\Documents\Diplomatiki_NAO\vosk-model-el-gr-0.7`.

Εγκατάσταση Python εξαρτήσεων (virtualenv):

```powershell
& .venv\Scripts\Activate.ps1
pip install sounddevice vosk python-dotenv
```

Ρύθμιση `.env` (προτεινόμενο -- το `App/main.py` φορτώνει `.env` αυτόματα αν υπάρχει):
Δημιούργησε στο φάκελο `nao_ai_integration` (ένα επίπεδο πάνω από `App`) αρχείο `.env` με:

```
VOSK_MODEL_PATH=C:\Users\frag2\Documents\Diplomatiki_NAO\vosk-model-el-gr-0.7
SOUNDDEVICE_DEVICE=30
STT_VOCAB=διπαε,δι.πα.ε,παπαδοπουλος,νικολαου
STT_VOCAB_DIR=C:\Users\frag2\Documents\Diplomatiki_NAO\controllers\nao_ai_integration\App\stt_vocab
STT_VOCAB_PROFILE=all
```

`SOUNDDEVICE_DEVICE` είναι προαιρετικό και μπορεί να είναι ο αριθμητικός index συσκευής εισόδου που βρήκες με `sounddevice.query_devices()` (βλέπε παρακάτω). Αν δεν το ορίσεις, το script χρησιμοποιεί την προεπιλεγμένη είσοδο του συστήματος.
`STT_VOCAB` είναι προαιρετικό και δέχεται custom λέξεις/ονόματα χωρισμένα με κόμμα για VOSK vocabulary injection.
`STT_VOCAB_DIR` είναι προαιρετικό και δείχνει σε φάκελο λεξιλογίων με αρχεία `common.txt`, `professors.txt`, `courses.txt`.
`STT_VOCAB_PROFILE` ορίζει ποιο group θα φορτώνεται: `all`, `professor`, `course`, `common`.
`STT_VOCAB_FILE` παραμένει διαθέσιμο για backward compatibility (single `.txt`/`.json` αρχείο).

Προτεινόμενη οργάνωση:

```
App/
  stt_vocab/
    common.txt
    professors.txt
    courses.txt
```

Έλεγχος μικροφώνου / index συσκευής:

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Βρες τον index για το μικρόφωνο που θέλεις (π.χ. `1` ή `30`) και όρισε `SOUNDDEVICE_DEVICE` στο `.env` ή ως περιβαλλοντική μεταβλητή στη σύνοδο του τερματικού.

Τρέξιμο με φωνητική είσοδο:

```powershell
# (ενεργό virtualenv)
python .\App\main.py
```

- Στο μενού του προγράμματος διάλεξε `v` για φωνή.
- Το push-to-talk flow: Πάτα Enter για εκκίνηση εγγραφής, ξαναπάτα Enter για τερματισμό. Η μεταγραφή εμφανίζεται και αποστέλλεται στο RAG.

Πιθανά σφάλματα & λύσεις

- Αν το πρόγραμμα εκτυπώνει "Φωνητική είσοδος μη διαθέσιμη", έλεγξε:
  - Ότι `vosk` είναι εγκατεστημένο στο ίδιο virtualenv που τρέχεις (`pip show vosk`).
  - Ότι `VOSK_MODEL_PATH` δείχνει στον ριζικό φάκελο του μοντέλου (περιέχει υποφακέλους όπως `am`, `conf`, `model`, κ.λπ.).
  - Ότι η `SOUNDDEVICE_DEVICE` είναι έγκυρος index (ή αφαίρεσέ τη για να χρησιμοποιήσεις default input).

Debug: δοκίμασε το παρακάτω για να φορτώσεις απευθείας το μοντέλο και να δεις αν υπάρχει σφάλμα:

```powershell
python -c "from vosk import Model; m=Model(r'C:\Users\frag2\Documents\Diplomatiki_NAO\vosk-model-el-gr-0.7'); print('model OK')"
```

Αν εμφανιστεί exception, κόλλησέ το εδώ και θα σε βοηθήσω να το διορθώσουμε.

## Πρόσθετες προτάσεις

- `requirements.txt`: μπορείς να προσθέσεις `sounddevice vosk python-dotenv` για να κάνεις εγκατάσταση με `pip install -r requirements.txt`.
- `env.example`: ένα παράδειγμα `.env` με `VOSK_MODEL_PATH` και `SOUNDDEVICE_DEVICE` βοηθάει άλλους χρήστες.

---

Σημείωση: οι παραπάνω οδηγίες αφορούν την τοπική offline φωνητική είσοδο. Αν θες, μπορώ να προσθέσω αυτόματα `requirements.txt` και `env.example` στο repo.

## Εκτέλεση του `App/main.py`

Το κύριο script είναι [App/main.py](App/main.py). Όταν τρέχεις το πρόγραμμα, εμφανίζεται μενού επιλογών εισόδου:

- `k`: πληκτρολόγιο — πληκτρολογείς το κείμενο απευθείας.
- `v`: φωνή (push-to-talk) — χρησιμοποιεί offline VOSK + `sounddevice` για εγγραφή και μεταγραφή.
- `n`: NAO (remote) — ξεκινάει απομακρυσμένη εγγραφή στον ρομπότ μέσω `ssh`, κατεβάζει το WAV με `scp` και το μεταγράφει τοπικά.
- `q`: έξοδος

Το πρόγραμμα παίρνει το παραγόμενο `user_text` και το στέλνει στη RAG engine (`rag_engine.answer_query()`), μορφώνει την απάντηση και την εκφωνεί με `nao_say()`.

### Σημαντικές μεταβλητές περιβάλλοντος που επηρεάζουν το `main.py`

- `RAG_MODEL` : override για το RAG model (π.χ. `gpt-4o-mini`).
- `RAG_TOP_K` : αριθμός κορυφαίων chunks για RAG (default `5`).
- `RAG_PRELOAD_RETRIEVAL_ON_BOOT` : αν είναι `1/true`, κάνει preload το embedding model + retrieval index στο startup (default `1`).
- `RAG_DEBUG_CHUNKS` : ενεργοποιεί εκτύπωση chunks (0/1).
- `VOSK_MODEL_PATH` : path στον φάκελο του VOSK μοντέλου (π.χ. `C:\...\vosk-model-el-gr-0.7`).
- `SOUNDDEVICE_DEVICE` : (προαιρετικό) index της εισόδου ήχου για `sounddevice`.
- `STT_VOCAB` : (προαιρετικό) comma-separated custom λέξεις/ονόματα για VOSK grammar injection.
- `STT_VOCAB_DIR` : (προαιρετικό) φάκελος με `common.txt`, `professors.txt`, `courses.txt`.
- `STT_VOCAB_PROFILE` : (προαιρετικό) `all|professor|course|common`.
- `STT_VOCAB_FILE` : (προαιρετικό) path σε `.txt`/`.json` με custom vocabulary για VOSK.
- `NAO_USER` : χρήστης για SSH στο NAO (default `nao`).
- `NAO_IP` / `NAO_PORT` : διεύθυνση/θύρα του NAO — μπορούν να δοθούν και ως CLI args `--nao-ip`/`--nao-port`.
- `NAO_RECORD_SEC` : πόσα δευτερόλεπτα θα εγγράψει ο NAO (default `5`).
- `NAO_PREP_SEC` : προ-προετοιμασία σεδ (countdown) πριν το remote recording (default `3`).
- `PUSH_MAX_SECONDS` : μέγιστη διάρκεια για το push-to-talk (mode `v`). Προεπιλογή `15` δευτερόλεπτα. Η εγγραφή θα τερματιστεί αυτόματα μετά από αυτό το όριο.
- `STT_BACKEND` : επιλέγει backend για offline STT (`vosk` ή `whispercpp`). Προεπιλογή `vosk`.
- `WHISPERCPP_BIN` : διαδρομή στο εκτελέσιμο `whisper.cpp` (όταν `STT_BACKEND=whispercpp`).
- `WHISPERCPP_MODEL` : διαδρομή στο ggml μοντέλο για `whisper.cpp`.

### Προαπαιτούμενα για κάθε mode

- `v` (φωνή): απαιτεί εγκατάσταση `vosk` και `sounddevice` και διαθέσιμο το μοντέλο στο `VOSK_MODEL_PATH`. Συνιστώ επίσης `soundfile`, `numpy`, `scipy` για fallback μετατροπές.
- `n` (NAO): απαιτεί SSH πρόσβαση στον NAO, το helper script `/home/nao/nao_record.py` στο ρομπότ και `scp` για μεταφορά αρχείου. Το helper χρησιμοποιεί Python2 στο ρομπότ.

### Debugging & συμπεριφορά σφαλμάτων

- Αν η μεταγραφή από NAO είναι κενή, το αρχείο WAV αποθηκεύεται σε `Tests/nao_last.wav` για επιθεώρηση.
- Αν το `VOSK` ή το μοντέλο δεν είναι διαθέσιμα, το `v` mode θα εμφανίσει προειδοποίηση και θα προτείνει χρήση `k`.
- Αν δεν υπάρχει `ffmpeg` και το αρχείο ήχου δεν είναι απλό WAV, το πρόγραμμα θα προτείνει την εγκατάσταση των Python libs (`soundfile numpy scipy`) ή `ffmpeg`.

### Όριο push-to-talk

- Το `v` mode (push-to-talk) σταματάει όταν πατήσεις ξανά Enter ή όταν φτάσει το όριο `PUSH_MAX_SECONDS` (default `15s`). Μπορείς να αλλάξεις `PUSH_MAX_SECONDS` στο `.env` ή στο περιβάλλον πριν τρέξεις το script.

### Whisper.cpp (προαιρετικά)

- Αν θέλεις να χρησιμοποιήσεις `whisper.cpp` για την μεταγραφή αρχείων (π.χ. για καλύτερη ποιότητα στα Ελληνικά), όρισε `STT_BACKEND=whispercpp` και ρύθμισε `WHISPERCPP_BIN` και `WHISPERCPP_MODEL` στο `.env`. Το `main.py` θα καλέσει το εκτελέσιμο για να κάνει το local transcription των WAV αρχείων που λαμβάνονται από τον NAO.

### Παραδείγματα εκτέλεσης

PowerShell (προσωρινά για το τρέχον session):

```powershell
$env:NAO_IP="192.168.-.-"
$env:VOSK_MODEL_PATH="C:\Users\frag2\Documents\Diplomatiki_NAO\vosk-model-el-gr-0.7"
python .\App\main.py
```

CMD (προσωρινά για την τρέχουσα κονσόλα):

```cmd
set NAO_IP=192.168.1.42
set VOSK_MODEL_PATH=C:\Users\frag2\Documents\Diplomatiki_NAO\vosk-model-el-gr-0.7
python App\main.py
```

Σημείωση: μπορείς επίσης να περάσεις `--nao-ip` και `--nao-port` στο `python App/main.py` για προσωρινή παράκαμψη.
Για το v ωστε να βρισκει το μικροφωνο : python -c "import sounddevice as sd; print(sd.query_devices())"
