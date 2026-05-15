# RAG Routing — Κατευθυντήριες και Decision Tree

## Σκοπός

Αυτό το έγγραφο περιγράφει τους κανόνες routing για το RAG backend του chatbot: πότε κάνουμε DB lookup, πότε document (vector) retrieval, πότε hybrid, και πώς συναρμολογούμε context για το Ollama. Είναι γραμμένο για να το επεξεργαστείς βημα-βημα και να αποφασίσουμε μαζί τις τελικές παραμέτρους.

## Γενικές αρχές

- DB-first: Δομημένα records (καθηγητές, μαθήματα, γραφεία, κωδικοί μαθημάτων, πιστωτικές μονάδες) πρέπει πρώτα να ελέγχονται στη βάση `NAO_IEE_IHU_DB.db`.
- Documents: Πολυκείμενα (οδηγοί σπουδών, προγράμματα, υποδομές) αναζητούνται με vector retrieval πάνω σε προ-επεξεργασμένα chunks.
- Hybrid: Όταν υπάρχει structured entity αλλά χρειάζονται επιπλέον λεπτομέρειες ή τεκμηρίωση, συνδυάζουμε DB summary + top document chunks.
- Online augmentation: Ενεργοποιείται μόνο για live/χρονικά ευαίσθητες ερωτήσεις και με whitelist domains.
- Safety-first: Αν το σύστημα δεν βρει αξιόπιστη πληροφορία, να απαντά "Δεν γνωρίζω" ή να ρωτά για follow-up, όχι να εκτελεί hallucinate.

## Απαραίτητα metadata σε κάθε τεμάχιο (chunk) ή DB record

- `source_name` (π.χ. Οδηγός_Σπουδών_24-25)
- `source_type` (DB | Document | Online)
- `path_or_id` (αρχείο ή πίνακας/γραμμή DB)
- `lang` (el-GR)
- `created_at` / `updated_at` (όπου υπάρχει)
- `topic_tags` (π.χ. course, faculty, infra)

## Decision Tree (βήμα-βήμα)

1. Query normalization
   - Καθαρισμός input (normalization: lowercasing, remove punctuation noise, expand abbreviations).
   - Εξαγωγή entities & intent (person, course, office, program, transport, credit, phd, erasmus).

2. DB lookup (fast path)
   - Αν intent==person OR intent==course OR intent==office OR intent==credit:
     - Κάνε exact/partial search στη DB (`name LIKE %...%` or `code = ...`).
     - Αν υπάρχει exact match και confidence >= DB_MATCH_BOOST_THRESHOLD → επιστροφή DB answer.
     - Αν υπάρχει partial match → proceed to Document-augmentation (βήμα 3).

3. Document retrieval (vector path)
   - Κάνε query embeddings + nearest neighbour search (k = K_DEFAULT).
   - Apply similarity threshold (SIMILARITY_THRESHOLD). Keep top `max_context_chunks` μετά dedup.
   - Re-rank chunks με keyword-overlap και metadata match (π.χ. topic_tags).
   - Αν τελικό score >= RETRIEVAL_CONFIDENCE_THRESHOLD → assemble context (top N chunks).
   - Αν όχι → fallback to Ollama with safety system prompt (να παραδεχτεί άγνοια).

4. Hybrid combine
   - Αν DB partial match + document chunks →
     - Build concise DB summary (1–2 προτάσεις) + attach top doc chunks as evidence.
     - Call Ollama to produce a natural, human-style answer using DB summary + doc context.

5. Online augmentation
   - Allowed only if query classified as `live` (ώρες, ανακοινώσεις, deadlines).
   - Query external whitelist domains; validate source; if recency/credibility ok → include as chunk with `source_type=Online` + timestamp.
   - Rate-limit and cache online results.

6. Post-process & respond
   - Use `preferred_output_length` from evaluation template to format final answer (short|long|voice).
   - Attach citations list: for each used chunk add `(source_name — short path — confidence)`.
   - For voice: shorten and ask if user θέλει link/πηγή.

## Παραμέτροι (προτεινόμενες, παραμετροποιήσιμες)

- `K_DEFAULT = 5` # πόσα nearest neighbours για αρχική ανάκτηση
- `max_context_chunks = 3` # πόσα chunks να συμπεριληφθούν στο prompt
- `SIMILARITY_THRESHOLD = 0.65` # αρχική αποδεκτή ομοιότητα
- `RETRIEVAL_CONFIDENCE_THRESHOLD = 0.6`
- `DB_MATCH_BOOST_THRESHOLD = 0.8` # αν DB match >= αυτό → direct DB answer
- `DEDUP_OVERLAP = 0.8` # αν δύο chunks έχουν >80% overlap → drop το δεύτερο

## Re-ranking recipe (προτεινόμενο)

score = w1 _ similarity_score + w2 _ keyword*overlap + w3 * metadata*match + w4 * recency_boost
προτεινόμενοι βάρη: w1=0.6, w2=0.2, w3=0.15, w4=0.05

## Prompt template (συνοπτικό)

System message:

- Ρόλος: "Είσαι βοηθητικός, φυσικός, μιλάς Ελληνικά. Απάντησε μόνο βάσει του παρεχόμενου context. Αν δεν υπάρχει επαρκής πληροφορία, πες 'Δεν γνωρίζω'. Τέλος: δώσε σύντομη λίστα πηγών. Μήκος απάντησης: {preferred_output_length}."

User payload (παράδειγμα):

- QUERY: "{user_query}"
- CHUNKS:
  - [SOURCE: {source_name} | path: {path_or_id} | score: {score}]
  - {chunk_text}

- INSTRUCTION: "Χρησιμοποίησε μόνο τα παραπάνω. Απάντησε φυσικά στα Ελληνικά. Μήκος: {preferred_output_length}. Δώσε σύντομη λίστα πηγών στο τέλος."

## Integration points (πού να τοποθετηθεί στο repo)

- Routing decision & orchestration: `App/main.py` και `App/voice_dialog_online_sd.py` (αντικαθιστούμε το direct `ollama_generate` path με call σε `rag_orchestrator.query(...)`).
- Κοινά helpers: `App/nao_common.py` ή νέο module `App/rag_backend.py` για: classifier, db_router, retriever_wrapper, re-ranker, prompt builder.
- Evaluation template: `App/rag_evaluation_template.csv` — χρησιμοποιείται για gold answers / tests.

## Checklist για ρυθμίσεις που πρέπει να αποφασίσουμε μαζί

- [ ] Τελικά τιμές: `SIMILARITY_THRESHOLD`, `K_DEFAULT`, `max_context_chunks`.
- [ ] Whitelist domains για online augmentation.
- [ ] Policy για `preferred_output_length` default (voice vs text).
- [ ] Πώς διαχειριζόμαστε conflicting DB vs doc info (DB authoritative? timestamp?).

## Παραδείγματα ροών (σύντομα)

- Ερώτηση: "Ποιος είναι ο Ευκλείδης Κεραμόπουλος?"
  - classifier -> intent=person -> DB lookup -> exact match -> return DB `short_answer` (ή `full_answer` αν preferred=long).

- Ερώτηση: "Πες μου για την πρακτική άσκηση"
  - classifier -> intent=program/policy -> Document retrieval -> top chunks -> re-rank -> assemble context -> Ollama -> return natural answer + citations.

- Ερώτηση: "Έχει βγει ανακοίνωση για την εγγραφή;"
  - classifier -> intent=live -> online augmentation -> fetch from whitelisted site(s) -> include if recency ok -> else ask user to check site.
