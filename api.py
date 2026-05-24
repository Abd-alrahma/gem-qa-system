"""
GEM Arabic QA API — v1.0
=========================
Extractive and Generative QA over the Arabic GEM website content.

  extractive — AraElectra finds the exact answer span   (fast, precise)
  generative — merges top-3 relevant chunks             (richer answer)
  auto       — picks mode based on confidence score

Run:   uvicorn api:app --reload
Needs: chunks.json + faiss_index.index  (built by scraper.py + build_index.py)
"""

import json
import re
import numpy as np
import faiss
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── App ───────────────────────────────────────────────────
app = FastAPI(title="GEM Arabic QA API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIDENCE_THRESHOLD = 0.25
RELEVANCE_THRESHOLD  = 0.42
MIN_COMBINED_RELEVANCE = 0.28
LOW_QA_CONFIDENCE = 0.35
DIRECT_CONFIDENCE = 0.88

# ── Load models ───────────────────────────────────────────
print("Loading models…")
DEVICE = 0 if torch.cuda.is_available() else -1

# Multilingual embedding — same model used in build_index.py
embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

# Arabic extractive QA model — fine-tuned on Arabic SQuAD
extractive_qa = pipeline(
    "question-answering",
    model="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA",
    tokenizer="ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA",
    device=DEVICE,
)

index = faiss.read_index("faiss_index.index")
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Ready — {len(chunks)} chunks, {index.ntotal} vectors")


# ── Helpers ───────────────────────────────────────────────
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
TOKEN_RE = re.compile(r"[\u0621-\u064Aa-zA-Z0-9]+")

ARABIC_STOPWORDS = {
    "في", "من", "على", "عن", "الى", "إلى", "ما", "ماذا", "هل", "كيف", "كم",
    "أين", "اين", "متى", "هو", "هي", "هذا", "هذه", "ذلك", "تلك", "و", "او",
    "أو", "ثم", "مع", "كما", "كان", "كانت", "يكون", "يمكن", "التي", "الذي",
    "الذين", "اللاتي", "أن", "ان", "إن", "فيه", "فيها", "له", "لها", "كل",
    "بعد", "قبل", "عند", "بين", "ضمن", "حول", "غير", "لقد", "قد",
}

QUERY_SYNONYMS = {
    "مواعيد": {"اوقات", "ساعات", "العمل", "فتح", "غلق"},
    "اوقات": {"مواعيد", "ساعات", "العمل", "فتح", "غلق"},
    "ساعات": {"مواعيد", "اوقات", "العمل", "فتح", "غلق"},
    "تذكره": {"تذاكر", "التذاكر", "شراء", "الشراء", "حجز"},
    "تذاكر": {"تذكره", "التذاكر", "شراء", "الشراء", "حجز"},
    "اشتري": {"شراء", "الشراء", "تذاكر", "تذكره", "حجز"},
    "احجز": {"حجز", "تذاكر", "تذكره", "شراء"},
    "سعر": {"اسعار", "التذاكر", "رسوم"},
    "اسعار": {"سعر", "التذاكر", "رسوم"},
    "مكان": {"موقع", "يقع", "عنوان", "الوصول"},
    "يقع": {"مكان", "موقع", "عنوان", "الوصول"},
    "العنوان": {"عنوان", "موقع", "يقع", "الوصول"},
}

def is_arabic(text: str) -> bool:
    return len(ARABIC_RE.findall(text)) > len(text) * 0.15


def normalize_arabic(text: str) -> str:
    text = DIACRITICS_RE.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    return text.lower()


def stem_token(token: str) -> str:
    if token.startswith("ال") and len(token) > 4:
        token = token[2:]
    for suffix in ("هما", "كما", "يات", "ات", "ون", "ين", "ها", "هم", "كم", "نا", "ه", "ة"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[:-len(suffix)]
            break
    return token


def tokenize(text: str) -> list[str]:
    normalized = normalize_arabic(text)
    tokens = TOKEN_RE.findall(normalized)
    clean_tokens = []
    for token in tokens:
        if len(token) <= 1 or token in ARABIC_STOPWORDS:
            continue
        clean_tokens.append(stem_token(token))
    return clean_tokens


def expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(stem_token(normalize_arabic(t)) for t in QUERY_SYNONYMS.get(token, set()))
    return expanded


def lexical_score(question: str, chunk: str) -> float:
    original_q_tokens = set(tokenize(question))
    q_tokens = expand_query_tokens(original_q_tokens)
    if not q_tokens:
        return 0.0
    c_tokens = set(tokenize(chunk))
    overlap = len(q_tokens & c_tokens) / max(len(original_q_tokens), 1)
    overlap = min(1.0, overlap)

    q_phrase = " ".join(tokenize(question))
    c_phrase = " ".join(tokenize(chunk))
    phrase_bonus = 0.15 if q_phrase and q_phrase in c_phrase else 0.0
    return min(1.0, overlap + phrase_bonus)


def has_any(text: str, words: set[str]) -> bool:
    tokens = set(tokenize(text))
    return bool(tokens & {stem_token(normalize_arabic(w)) for w in words})


def find_chunk_containing(*phrases: str) -> str:
    for chunk in chunks:
        normalized = normalize_arabic(chunk)
        if all(normalize_arabic(phrase) in normalized for phrase in phrases):
            return chunk
    return ""


def make_direct_response(answer: str, context: str = "") -> dict:
    return {
        "mode": "extractive",
        "answer": answer,
        "confidence": DIRECT_CONFIDENCE,
        "context": context,
        "retrieval_score": 1.0,
        "answer_type": "direct",
    }


SUGGESTED_QA = [
    {
        "category": "زيارة المتحف",
        "question": "أين يقع المتحف المصري الكبير؟",
        "answer": "يقع المتحف المصري الكبير على طريق القاهرة - الإسكندرية الصحراوي، ميدان الرماية، الجيزة، مصر، ١٢١١١.",
        "context_phrase": "يقع المتحف المصري الكبير",
    },
    {
        "category": "زيارة المتحف",
        "question": "ما هي ساعات عمل المتحف؟",
        "answer": "يعمل مجمع المتحف يوميًا عدا السبت والأربعاء من 8:30 صباحًا إلى 7 مساءً، وتعمل قاعات العرض من 9 صباحًا إلى 6 مساءً. يومي السبت والأربعاء يعمل المجمع من 8:30 صباحًا إلى 10 مساءً، وتعمل قاعات العرض من 9 صباحًا إلى 9 مساءً.",
        "context_phrase": "ســاعات الـــعمل",
    },
    {
        "category": "زيارة المتحف",
        "question": "متى آخر موعد لشراء التذاكر؟",
        "answer": "آخر موعد لشراء التذاكر هو 5 مساءً في الأيام العادية، و8 مساءً يومي السبت والأربعاء.",
        "context_phrase": "آخر موعد لشراء التذاكر",
    },
    {
        "category": "زيارة المتحف",
        "question": "ما هي مواعيد متحف الأطفال؟",
        "answer": "متحف الأطفال يعمل من الأحد إلى الخميس من 1 مساءً إلى 5 مساءً، ويومي الجمعة والسبت من 10 صباحًا إلى 5 مساءً.",
        "context_phrase": "متحف الأطفال الأحد إلى الخميس",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "كيف أشتري تذكرة؟",
        "answer": "وفقًا للأسئلة الشائعة، يجب حجز التذاكر مسبقًا عبر موقع المتحف مع اختيار موعد محدد للدخول. واعتبارًا من 1 ديسمبر 2025، أصبح الحجز عبر موقع المتحف هو الطريقة الوحيدة لشراء التذاكر.",
        "context_phrase": "الطريقة الوحيدة لشراء التذاكر",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "هل يجب حجز التذاكر مسبقًا؟",
        "answer": "نعم، يجب حجز التذاكر مسبقًا عبر موقع المتحف مع اختيار موعد محدد للدخول.",
        "context_phrase": "يجب حجز التذاكر مسبقًا",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "هل يمكن استرداد قيمة التذاكر؟",
        "answer": "لا يمكن استرداد قيمة التذاكر بعد شرائها، إلا في حالات الضرورة التي تقدرها إدارة المتحف.",
        "context_phrase": "لا يمكن استرداد قيمة التذاكر",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "هل يوجد دخول مجاني لبعض الفئات؟",
        "answer": "نعم، يُمنح الدخول المجاني لفئات محددة مع ضرورة إبراز بطاقة هوية سارية لإثبات الاستحقاق، مع ملاحظة أن رسوم الجولات الإرشادية تُدفع منفصلة.",
        "context_phrase": "يُمنح الدخول المجاني",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "هل المتحف مجهز لذوي الهمم؟",
        "answer": "نعم، المتحف مجهز بممرات ومنحدرات ومصاعد ومسارات واسعة لضمان زيارة مريحة وسهلة، ويمكن للزوار الذين يحتاجون مساعدة التوجه إلى مكتب الاستعلامات.",
        "context_phrase": "المتحف مجهز بممرات",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "هل يوجد موقف سيارات؟",
        "answer": "نعم، يوجد موقف سيارات مقابل رسوم، وينصح المتحف بالقدوم بسيارة مشتركة وقت الذروة لتجنب الازدحام.",
        "context_phrase": "يوجد موقف سيارات",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "هل يوجد مصلى داخل المتحف؟",
        "answer": "نعم، يوجد بالدور الأرضي مصلى من غرفتين؛ غرفة للسيدات وغرفة للرجال، مع وجود ميضأة.",
        "context_phrase": "مصلى من غرفتين",
    },
    {
        "category": "التذاكر والخدمات",
        "question": "هل توجد خزائن أو أمانات؟",
        "answer": "نعم، تتوفر خدمة الأمانات والخزائن مجانًا، ويمكن طلبها من مكتب الاستعلامات.",
        "context_phrase": "الأمانات والخزائن مجانًا",
    },
    {
        "category": "الجولات",
        "question": "هل يقدم المتحف جولات إرشادية؟",
        "answer": "نعم، يقدم المتحف المصري الكبير جولات إرشادية باللغتين العربية والإنجليزية.",
        "context_phrase": "جولات إرشادية",
    },
    {
        "category": "الجولات",
        "question": "ما مدة الجولة الإرشادية؟",
        "answer": "مدة الجولة الإرشادية ساعتان، وتشمل قاعات توت عنخ آمون، والبهو العظيم، والدرج العظيم، والقاعات الرئيسة.",
        "context_phrase": "الجولة الإرشادية مدتها",
    },
    {
        "category": "الجولات",
        "question": "كيف أحجز جولة خاصة؟",
        "answer": "يمكن ترتيب جولة خاصة مخصصة حسب اللغة والاهتمامات وسرعة التجول، وللحجز والاستفسار يرجى التواصل مع فريق الحجز عبر booking@gem.eg.",
        "context_phrase": "booking@gem.eg",
    },
    {
        "category": "الجولات",
        "question": "ما هي جولة خارج أوقات العمل؟",
        "answer": "جولة خارج أوقات العمل تجربة مسائية هادئة داخل المتحف بعد ساعات العمل الرسمية، ويُشترط لها الحجز المسبق.",
        "context_phrase": "جولة خارج أوقات العمل",
    },
    {
        "category": "المجموعات الأثرية",
        "question": "كم عدد القطع الأثرية في الدرج العظيم؟",
        "answer": "يضم الدرج العظيم ٥٩ قطعة أثرية مذهلة.",
        "context_phrase": "٥٩ قطعة أثرية",
    },
    {
        "category": "المجموعات الأثرية",
        "question": "كم عدد قاعات العرض الرئيسية؟",
        "answer": "تضم قاعات العرض الرئيسية اثنتي عشرة قاعة منسقة بعناية تعرض قصة الحضارة المصرية من عصور ما قبل التاريخ حتى العصرين اليوناني والروماني.",
        "context_phrase": "اثنتا عشرة قاعة",
    },
    {
        "category": "المجموعات الأثرية",
        "question": "كم قطعة تعرض في قاعات توت عنخ آمون؟",
        "answer": "تعرض قاعات توت عنخ آمون أكثر من ٥٠٠٠ قطعة أثرية من مقبرة الملك الشاب مجتمعة لأول مرة.",
        "context_phrase": "أكثر من ٥٠٠٠ قطعة أثرية",
    },
    {
        "category": "المجموعات الأثرية",
        "question": "ما هي أهم القطع الأثرية في المتحف؟",
        "answer": "من أهم القطع الأثرية: قناع الدفن الذهبي للملك توت عنخ آمون، تمثال صقر، مسلة الملك رمسيس الثاني، نموذج قارب لأوخ-حتب، تمثال المعبود بتاح والملك رمسيس الثاني مع المعبودة سخمت، قمة مسلة للملكة حتشبسوت، كرسي العرش الذهبي، وقناع مومياء لمسحتي.",
        "context_phrase": "أهم القطع الأثرية",
    },
    {
        "category": "قطع مختارة",
        "question": "ما رقم أثر قناع الدفن الذهبي؟",
        "answer": "رقم أثر قناع الدفن الذهبي للملك توت عنخ آمون هو 8.",
        "context_phrase": "قناع الدفن الذهبي",
    },
    {
        "category": "قطع مختارة",
        "question": "ما مادة صنع قناع الدفن الذهبي؟",
        "answer": "صُنع قناع الدفن الذهبي من ذهب وزجاج ولازورد وأوبسيديان وعقيق وفاينس وكوارتزيت.",
        "context_phrase": "قناع الدفن الذهبي",
    },
    {
        "category": "قطع مختارة",
        "question": "ما رقم أثر كرسي العرش الذهبي؟",
        "answer": "رقم أثر كرسي العرش الذهبي هو 4573.",
        "context_phrase": "كرسي العرش الذهبي",
    },
    {
        "category": "قطع مختارة",
        "question": "ما رقم أثر مسلة الملك رمسيس الثاني؟",
        "answer": "رقم أثر مسلة الملك رمسيس الثاني هو 21331.",
        "context_phrase": "مسلة الملك رمسيس الثاني",
    },
    {
        "category": "قطع مختارة",
        "question": "ما مادة صنع مسلة الملك رمسيس الثاني؟",
        "answer": "صُنعت مسلة الملك رمسيس الثاني من الجرانيت.",
        "context_phrase": "مسلة الملك رمسيس الثاني",
    },
    {
        "category": "قطع مختارة",
        "question": "ما رقم أثر تمثال صقر؟",
        "answer": "رقم أثر تمثال صقر هو 2375.",
        "context_phrase": "تمثال صقر",
    },
    {
        "category": "الأطفال والتعليم",
        "question": "أين يقع متحف الأطفال؟",
        "answer": "يقع متحف الأطفال في قلب المتحف المصري الكبير.",
        "context_phrase": "يقع متحف الأطفال",
    },
    {
        "category": "الأطفال والتعليم",
        "question": "ما الفئة العمرية لبرامج الزيارات المدرسية؟",
        "answer": "برامج الزيارات المدرسية مخصصة للطلاب من عمر 6 إلى 16 عامًا.",
        "context_phrase": "من عمر 6 إلى 16 عامًا",
    },
    {
        "category": "الأطفال والتعليم",
        "question": "كيف أحجز زيارة مدرسية؟",
        "answer": "للحجز والاستفسار عن الزيارات المدرسية، يُرجى التواصل عبر البريد الإلكتروني learning@gem.eg.",
        "context_phrase": "learning@gem.eg",
    },
    {
        "category": "الخدمات",
        "question": "هل يوجد دليل صوتي؟",
        "answer": "نعم، يقدم المتحف دليلًا صوتيًا باللغات العربية والإنجليزية واليابانية، ويشمل أكثر من 100 محطة في خمس مناطق رئيسية بالمتحف.",
        "context_phrase": "دليلًا صوتيًا",
    },
    {
        "category": "الخدمات",
        "question": "هل توجد مطاعم أو مشروبات داخل المتحف؟",
        "answer": "نعم، يضم المتحف مجموعة متنوعة من منافذ الطعام والمشروبات ليستمتع الزوار بوجبات أو وجبات خفيفة ومشروبات خلال الزيارة.",
        "context_phrase": "منافذ الطعام والمشروبات",
    },
    {
        "category": "الخدمات",
        "question": "هل يُسمح بالتصوير داخل المتحف؟",
        "answer": "يُسمح بالتصوير الفوتوغرافي أو المرئي الشخصي وغير التجاري داخل المتحف، لكن يُمنع التصوير بالفلاش واستخدام الحوامل الثلاثية وعصي السيلفي والطائرات المسيرة ووحدات الإضاءة الخارجية وأجهزة البث المباشر.",
        "context_phrase": "يُسمح بالتصوير",
    },
]


CURATED_QA_BY_QUESTION = {
    normalize_arabic(item["question"]): item for item in SUGGESTED_QA
}


def curated_direct_answer(question: str) -> dict | None:
    item = CURATED_QA_BY_QUESTION.get(normalize_arabic(question))
    if not item:
        return None
    context = find_chunk_containing(item["context_phrase"]) if item.get("context_phrase") else ""
    return make_direct_response(item["answer"], context)


def sentence_with(chunk: str, *phrases: str) -> str:
    sentences = re.split(r"(?<=[.!؟])\s+|،\s+", chunk)
    normalized_phrases = [normalize_arabic(p) for p in phrases]
    for sentence in sentences:
        normalized = normalize_arabic(sentence)
        if all(p in normalized for p in normalized_phrases):
            return sentence.strip()
    return chunk.strip()


def readable_excerpt(text: str, max_words: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip("،؛:") + "..."


def first_relevant_sentences(question: str, chunk: str, max_sentences: int = 2) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!؟])\s+|(?<=،)\s+", chunk) if s.strip()]
    if not sentences:
        return readable_excerpt(chunk)

    ranked = sorted(
        sentences,
        key=lambda s: lexical_score(question, s),
        reverse=True,
    )
    chosen = [s for s in ranked[:max_sentences] if lexical_score(question, s) > 0]
    if not chosen:
        chosen = sentences[:max_sentences]
    return readable_excerpt(" ".join(chosen), max_words=90)


def generative_answer_from_parts(question: str, parts: list[str]) -> str:
    evidence = [
        first_relevant_sentences(question, part, max_sentences=2)
        for part in parts
    ]
    evidence = [e for e in evidence if e]
    if not evidence:
        return ""

    if len(evidence) == 1:
        return readable_excerpt(
            "بناءً على النص المسترجع من الموقع: " + evidence[0],
            max_words=150,
        )

    joined = " كما يوضح الموقع أن ".join(evidence)
    return readable_excerpt(
        "بناءً على أكثر من جزء مسترجع من الموقع، " + joined,
        max_words=180,
    )


def extract_artifact_title(chunk: str) -> str:
    match = re.search(r"قطعة أثرية:\s*(.+?)\.\s", chunk)
    return match.group(1).strip() if match else ""


def extract_labeled_field(chunk: str, label: str) -> str:
    labels = [
        "رقم الأثر", "المجموعة", "الحِقْبةُ", "الأسرة", "الوصف",
        "المَنشأ", "الإقليم", "المنطقة", "مادة الصنع", "الأبعاد",
        "الارتفاع", "العرض", "الطول", "الوزن", "القُطر",
    ]
    next_labels = "|".join(re.escape(x) for x in labels if x != label)
    pattern = rf"{re.escape(label)}:\s*(.+?)(?=\s+(?:{next_labels})(?::|\s)|$)"
    match = re.search(pattern, chunk)
    return match.group(1).strip() if match else ""


def artifact_direct_answer(question: str) -> dict | None:
    artifact_chunks = [chunk for chunk in chunks if chunk.startswith("قطعة أثرية:")]
    if not artifact_chunks:
        return None

    q_tokens = set(tokenize(question))
    ranked = []
    for chunk in artifact_chunks:
        title = extract_artifact_title(chunk)
        if not title:
            continue
        title_tokens = set(tokenize(title))
        if not title_tokens:
            continue
        title_overlap = len(q_tokens & title_tokens) / len(title_tokens)
        question_overlap = len(q_tokens & title_tokens) / max(len(q_tokens), 1)
        score = max(lexical_score(question, title), title_overlap, question_overlap)
        ranked.append((score, title, chunk))

    if not ranked:
        return None

    score, title, chunk = max(ranked, key=lambda item: item[0])
    if score < 0.45:
        return None

    field = ""
    if has_any(question, {"رقم", "اثر", "الأثر"}):
        field = extract_labeled_field(chunk, "رقم الأثر")
    elif has_any(question, {"مجموعة", "مكان", "قاعة", "قاعات"}):
        field = extract_labeled_field(chunk, "المجموعة")
    elif has_any(question, {"حقبة", "عصر", "تاريخ", "الدولة"}):
        field = extract_labeled_field(chunk, "الحِقْبةُ")
    elif has_any(question, {"اسرة", "الأسرة"}):
        field = extract_labeled_field(chunk, "الأسرة")
    elif has_any(question, {"مادة", "صنع", "مصنوع"}):
        field = extract_labeled_field(chunk, "مادة الصنع")
    elif has_any(question, {"ابعاد", "ارتفاع", "عرض", "طول", "وزن", "قطر"}):
        for label in ("الأبعاد", "الارتفاع", "العرض", "الطول", "الوزن", "القُطر"):
            value = extract_labeled_field(chunk, label)
            if value:
                field = f"{label}: {value}"
                break

    answer = f"{title}: {field}" if field else readable_excerpt(chunk, max_words=95)
    return {
        "mode": "extractive",
        "answer": answer,
        "confidence": 0.95,
        "context": chunk,
        "retrieval_score": round(score, 4),
        "answer_type": "artifact_direct",
    }


def direct_answer(question: str) -> dict | None:
    """
    Deterministic answers for common website questions. These prevent the
    extractive model from returning a tiny but irrelevant span.
    """
    q = normalize_arabic(question)

    if (
        has_any(q, {"أين", "اين", "يقع", "مكان", "العنوان", "موقع"})
        and has_any(q, {"المتحف", "متحف", "المصري", "الكبير"})
    ):
        chunk = find_chunk_containing("يقع المتحف المصري الكبير")
        if chunk:
            answer = re.sub(r"^الوصول إلى المتحف\s*", "", chunk).strip()
            return {
                "mode": "extractive",
                "answer": answer,
                "confidence": DIRECT_CONFIDENCE,
                "context": chunk,
                "retrieval_score": 1.0,
                "answer_type": "direct",
            }

    if has_any(q, {"مواعيد", "اوقات", "ساعات", "العمل", "يفتح", "غلق"}):
        chunk = find_chunk_containing("ساعات", "مجمع المتحف")
        if not chunk:
            chunk = find_chunk_containing("أوقات الزيارة")
        if chunk:
            return {
                "mode": "extractive",
                "answer": chunk,
                "confidence": DIRECT_CONFIDENCE,
                "context": chunk,
                "retrieval_score": 1.0,
                "answer_type": "direct",
            }

    if has_any(q, {"اشتري", "شراء", "احجز", "حجز", "تذكرة", "تذاكر", "التذاكر"}):
        chunk = find_chunk_containing("وسيلة الشراء")
        faq_chunk = find_chunk_containing("الطريقة الوحيدة لشراء التذاكر")
        if chunk:
            answer = (
                "وفقًا للأسئلة الشائعة، يجب حجز التذاكر مسبقًا عبر موقع المتحف مع اختيار "
                "موعد محدد للدخول. واعتبارًا من 1 ديسمبر 2025، أصبح الحجز عبر موقع المتحف "
                "هو الطريقة الوحيدة لشراء التذاكر."
            )
            return {
                "mode": "extractive",
                "answer": answer,
                "confidence": DIRECT_CONFIDENCE,
                "context": faq_chunk or chunk,
                "retrieval_score": 1.0,
                "answer_type": "direct",
            }

    if (
        any(word in q for word in ("عدد", "كم", "اجمالي"))
        and any(word in q for word in ("قطع", "قطعه", "اثري", "اثريه"))
    ):
        if "توت" in q or "عنخ" in q or "امون" in q:
            chunk = find_chunk_containing("أكثر من ٥٠٠٠ قطعة أثرية")
            if chunk:
                return {
                    "mode": "extractive",
                    "answer": "تعرض قاعات توت عنخ آمون أكثر من ٥٠٠٠ قطعة أثرية من مقبرة الملك الشاب مجتمعة لأول مرة.",
                    "confidence": DIRECT_CONFIDENCE,
                    "context": chunk,
                    "retrieval_score": 1.0,
                    "answer_type": "direct",
                }

        if "درج" in q or "الدَّرَج" in q or "العظيم" in q:
            chunk = find_chunk_containing("٥٩ قطعة أثرية")
            if chunk:
                return {
                    "mode": "extractive",
                    "answer": "يضم الدرج العظيم ٥٩ قطعة أثرية مذهلة.",
                    "confidence": DIRECT_CONFIDENCE,
                    "context": chunk,
                    "retrieval_score": 1.0,
                    "answer_type": "direct",
                }

        chunk = find_chunk_containing("٥٩ قطعة أثرية")
        if chunk:
            answer = (
                "النص المتاح لا يذكر إجمالي عدد كل القطع الأثرية في المتحف. "
                "لكنه يذكر أن الدرج العظيم يضم ٥٩ قطعة أثرية، وأن قاعات توت عنخ آمون "
                "تعرض أكثر من ٥٠٠٠ قطعة أثرية من مقبرة الملك."
            )
            return {
                "mode": "extractive",
                "answer": answer,
                "confidence": DIRECT_CONFIDENCE,
                "context": chunk,
                "retrieval_score": 1.0,
                "answer_type": "direct",
            }

    artifact = artifact_direct_answer(question)
    if artifact:
        return artifact

    return None


def retrieve(question: str, k: int) -> list[dict]:
    """
    Embed question with multilingual-e5 (query prefix required)
    and rerank with a light Arabic lexical score. Dense retrieval helps with
    synonyms, while lexical overlap keeps answers tied to the actual question.
    """
    prefixed = "query: " + question
    emb = embedding_model.encode([prefixed], normalize_embeddings=True)
    search_k = min(index.ntotal, max(k * 4, 20))
    D, I = index.search(np.array(emb, dtype="float32"), search_k)

    results = []
    for j, i in enumerate(I[0]):
        if i == -1:
            continue
        chunk = chunks[i]
        vector = float(D[0][j])
        lexical = lexical_score(question, chunk)
        combined = (0.70 * vector) + (0.30 * lexical)
        results.append({
            "chunk": chunk,
            "vector_score": vector,
            "lexical_score": lexical,
            "score": combined,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def is_good_answer(text: str) -> bool:
    """Reject answers that are too short or look like garbage."""
    if not text or len(text.split()) < 3:
        return False
    bad_fragments = {
        "لدخول واحد فقط",
        "مواعيد العمل الرسمية",
        "على طول نهر النيل",
    }
    if text.strip() in bad_fragments:
        return False
    if text.strip().endswith("؟"):   # answer ends with question mark → wrong
        return False
    return True


# ── EXTRACTIVE ────────────────────────────────────────────
def extractive_qa_fn(question: str, top_k: int = 5) -> dict:
    """
    Run AraElectra on all retrieved chunks.
    Return the span with the HIGHEST confidence score.
    Short spans (< 4 words) are rejected in favour of the next best.
    """
    direct = direct_answer(question)
    if direct:
        return direct

    results = retrieve(question, k=top_k)
    if not results:
        return {"mode": "extractive",
                "answer": "لم يتم العثور على إجابة.",
                "confidence": 0.0, "context": ""}

    if results[0]["score"] < MIN_COMBINED_RELEVANCE:
        return {
            "mode": "extractive",
            "answer": "لا توجد معلومات كافية في صفحات الموقع للإجابة على هذا السؤال.",
            "confidence": 0.0,
            "context": "",
            "retrieval_score": round(results[0]["score"], 4),
        }

    best = {
        "qa_score": 0.0,
        "answer": "",
        "context": "",
        "retrieval": results[0],
        "final_score": 0.0,
    }

    for item in results:
        chunk = item["chunk"]
        if item["score"] < MIN_COMBINED_RELEVANCE:
            continue
        try:
            r = extractive_qa(
                question=question,
                context=chunk,
                max_answer_len=150,
            )
            span = r["answer"].strip()
            # Reject spans that are too short or are question text
            if not is_good_answer(span):
                continue
            final_score = (0.65 * float(r["score"])) + (0.35 * item["score"])
            if final_score > best["final_score"]:
                best = {
                    "qa_score": float(r["score"]),
                    "answer": span,
                    "context": chunk,
                    "retrieval": item,
                    "final_score": final_score,
                }
        except Exception:
            continue

    # Fallback: if no good span found, return the most relevant sentences.
    if not best["answer"] or best["qa_score"] < LOW_QA_CONFIDENCE:
        best["answer"] = first_relevant_sentences(question, results[0]["chunk"], max_sentences=2)
        best["context"] = results[0]["chunk"]
        best["retrieval"] = results[0]
        best["qa_score"] = 0.0

    return {
        "mode":            "extractive",
        "answer":          best["answer"],
        "confidence":      round(best["qa_score"], 4),
        "context":         best["context"],
        "retrieval_score": round(best["retrieval"]["score"], 4),
        "vector_score":    round(best["retrieval"]["vector_score"], 4),
        "lexical_score":   round(best["retrieval"]["lexical_score"], 4),
    }


# ── GENERATIVE ────────────────────────────────────────────
def generative_qa_fn(question: str, top_k: int = 10) -> dict:
    """
    Merge the answer portions of the top relevant chunks into one
    flowing paragraph. Only chunks above RELEVANCE_THRESHOLD are used.
    Capped at 3 chunks to keep answers focused.
    """
    results = retrieve(question, k=top_k)
    if not results:
        return {"mode": "generative",
                "answer": "لا توجد معلومات كافية للإجابة على هذا السؤال.",
                "sources_used": 0}

    top_score = round(results[0]["score"], 4)
    if results[0]["score"] < MIN_COMBINED_RELEVANCE:
        return {
            "mode": "generative",
            "answer": "لا توجد معلومات كافية في صفحات الموقع للإجابة على هذا السؤال.",
            "sources_used": 0,
            "retrieval_score": top_score,
        }

    # Filter by relevance threshold
    relevant = [item for item in results if item["score"] >= RELEVANCE_THRESHOLD]
    if not relevant:
        relevant = [results[0]]   # always use at least the best chunk

    # Cap at 3 and deduplicate
    seen_keys, parts = set(), []
    for item in relevant[:3]:
        chunk = item["chunk"]
        key = chunk[:50]
        if key not in seen_keys:
            seen_keys.add(key)
            parts.append(chunk)

    # Build a fuller answer from several relevant sections.
    combined = generative_answer_from_parts(question, parts)
    combined = re.sub(r"\s+", " ", combined).strip()

    return {
        "mode":            "generative",
        "answer":          combined,
        "sources_used":    len(parts),
        "retrieval_score": top_score,
        "vector_score":    round(results[0]["vector_score"], 4),
        "lexical_score":   round(results[0]["lexical_score"], 4),
    }


# ── AUTO ──────────────────────────────────────────────────
def auto_qa_fn(question: str, top_k: int = 5) -> dict:
    ext = extractive_qa_fn(question, top_k)
    if ext["confidence"] >= CONFIDENCE_THRESHOLD:
        return {**ext,
                "note": f"وضع الاستخراج — الثقة {ext['confidence']:.0%}"}
    gen = generative_qa_fn(question, top_k)
    return {**gen,
            "note": f"وضع التوليد — ثقة الاستخراج كانت {ext['confidence']:.0%} فقط"}


# ── Endpoints ─────────────────────────────────────────────
@app.get("/ask")
def ask(
    question: str = Query(..., description="سؤالك باللغة العربية"),
    mode: str = Query(
        default="auto",
        enum=["extractive", "generative", "auto"],
        description=(
            "extractive = إجابة قصيرة دقيقة\n"
            "generative = إجابة مفصلة من عدة فقرات\n"
            "auto       = اختيار تلقائي حسب درجة الثقة"
        ),
    ),
    top_k: int = Query(default=5, ge=1, le=20),
):
    if mode == "extractive":
        return extractive_qa_fn(question, top_k)
    if mode == "generative":
        return generative_qa_fn(question, top_k)
    return auto_qa_fn(question, top_k)


@app.get("/suggested-questions")
def suggested_questions():
    return {
        "count": len(SUGGESTED_QA),
        "items": [
            {
                "category": item["category"],
                "question": item["question"],
                "answer": item["answer"],
            }
            for item in SUGGESTED_QA
        ],
    }


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "chunks":  len(chunks),
        "vectors": index.ntotal,
        "language": "Arabic (العربية)",
        "models": {
            "embedding":  "intfloat/multilingual-e5-base",
            "extractive": "ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA",
            "device":     "GPU" if DEVICE == 0 else "CPU",
        },
        "thresholds": {
            "confidence": CONFIDENCE_THRESHOLD,
            "relevance":  RELEVANCE_THRESHOLD,
            "minimum_combined_relevance": MIN_COMBINED_RELEVANCE,
        },
    }
