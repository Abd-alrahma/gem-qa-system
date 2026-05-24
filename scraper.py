"""
scraper.py — GEM Arabic Scraper v3 (Section-Based)
====================================================
Key improvement: chunks by SECTION (heading + its paragraphs),
not by word count. Each chunk = one topic = one coherent answer.

Run: python scraper.py
"""

import json, re, time, hashlib, sys
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────
BASE_URL    = "https://gem.eg/ar"
MAX_PAGES   = 100
PAGE_WAIT   = 1
OUTPUT      = "chunks.json"
MIN_WORDS   = 15    # minimum words to keep a chunk
MAX_WORDS   = 120   # maximum words per chunk before splitting

SEED_URLS = [
    "https://gem.eg/ar/",
    "https://gem.eg/ar/about/museum-story/",
    "https://gem.eg/ar/visit/plan-your-visit/",
    "https://gem.eg/ar/visit/plan-your-visit/opening-hours/",
    "https://gem.eg/ar/visit/plan-your-visit/visitor-tips/",
    "https://gem.eg/ar/faqs/",
    "https://gem.eg/ar/collection/",
    "https://gem.eg/ar/collection/artefacts/",
    "https://gem.eg/ar/whats-on/events/",
]

ARTIFACT_URLS = [
    "https://gem.eg/ar/collection/artefacts/the-golden-burial-mask-of-tutankhamun/",
    "https://gem.eg/ar/collection/artefacts/golden-throne/",
    "https://gem.eg/ar/collection/artefacts/royal-diadem/",
    "https://gem.eg/ar/collection/artefacts/canopic-chest/",
    "https://gem.eg/ar/collection/artefacts/canopic-coffinette/",
    "https://gem.eg/ar/collection/artefacts/canopic-coffinette-3/",
    "https://gem.eg/ar/collection/artefacts/outer-coffin/",
    "https://gem.eg/ar/collection/artefacts/guardian-statue-with-nemes-headcloth/",
    "https://gem.eg/ar/collection/artefacts/guardian-statue-with-khat-headdress/",
    "https://gem.eg/ar/collection/artefacts/festival-earrings/",
    "https://gem.eg/ar/collection/artefacts/lions-headrest/",
    "https://gem.eg/ar/collection/artefacts/cow-head/",
    "https://gem.eg/ar/collection/artefacts/model-of-funerary-boat/",
    "https://gem.eg/ar/collection/artefacts/isis-and-horus/",
    "https://gem.eg/ar/collection/artefacts/seated-male-figure/",
    "https://gem.eg/ar/collection/artefacts/statue-of-nesmin/",
    "https://gem.eg/ar/collection/artefacts/upper-part-of-a-royal-statue/",
    "https://gem.eg/ar/collection/artefacts/stick-of-mesehti/",
    "https://gem.eg/ar/collection/artefacts/ritual-carrying-box/",
    "https://gem.eg/ar/collection/artefacts/a-cup-treasure-of-tod/",
    "https://gem.eg/ar/collection/artefacts/small-bowl-with-handles-treasure-of-tod/",
    "https://gem.eg/ar/collection/artefacts/small-two-handled-bowl-treasure-of-tod/",
    "https://gem.eg/ar/collection/artefacts/a-string-with-carnelian-beads-treasure-of-tod/",
    "https://gem.eg/ar/collection/artefacts/vessel-with-gazelles/",
    "https://gem.eg/ar/collection/artefacts/faience-amulet-dendera-treasure-hoards/",
    "https://gem.eg/ar/collection/artefacts/a-gilt-hawk-figure-dendera-treasure-hoards/",
]

SEED_URLS.extend(ARTIFACT_URLS)

IGNORE_URL = ["/en/","javascript:","mailto:","#",".pdf",
              ".jpg",".png",".webp",".svg","/cdn-cgi/"]

# ── Noise: remove any chunk containing these strings ─────
NOISE = [
    "سياسة الخصوصية","ملفات تعريف الارتباط","الشروط والأحكام",
    "باستخدام موقعنا","تُقر بأنك","كلمات مُقترحة","ابدأ من هنا",
    "يرجى تفقد تطبيق","غير متوفرة بعد","سيتم إعادة توجيهك",
    "Buy tickets","اشتر تذكرتك","تابعنا على","فيسبوك","انستغرام",
    "اشتراك في النشرة","أدخل بريدك","حقوق الملكية",
    "جميع الحقوق محفوظة","cookie","Cookie","privacy","Privacy",
    "Skip to","تواصل معنا عبر","إرسال استفساركم",
    "يُرجى إرسال","للتواصل يُرْجى","نرحب باستفساراتكم",
    "فيما يتعلق بالتذاكر","متجر الهدايا",
    "مركز المؤتمرات","متحف الأطفال\nالأخبار",
    "ابدأ","بحث","موافق",
]

ARABIC = re.compile(r"[\u0600-\u06FF]")

def is_noise(text):
    for n in NOISE:
        if n in text:
            return True
    if len(ARABIC.findall(text)) < 8:
        return True
    return False

def clean(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def dedup(lst):
    seen, out = set(), []
    for x in lst:
        h = hashlib.md5(x.encode()).hexdigest()
        if h not in seen:
            seen.add(h); out.append(x)
    return out

def split_long(text, max_w=MAX_WORDS, overlap=10):
    """Only split if truly too long. Keeps semantic units together."""
    words = text.split()
    if len(words) <= max_w:
        return [text]
    parts, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+max_w])
        parts.append(chunk)
        i += max_w - overlap
    return parts

def clean_lines(lines):
    skip = {
        "مشاركة", "بحث", "ابدأ", "كلمات مُقترحة", "آت قريبًا",
        "مدونة السلوك", "سياسة ملفات تعريف الارتباط", "سياسة الخصوصية",
        "الشروط والأحكام", "الأسئلة الأكثر شيوعًا", "تواصل معنا",
    }
    out = []
    for line in lines:
        line = clean(line)
        if not line or line in skip:
            continue
        if len(ARABIC.findall(line)) < 2 and not re.search(r"\d", line):
            continue
        if out and out[-1] == line:
            continue
        out.append(line)
    return out

def extract_artifact_chunk(root, url):
    """Build one structured chunk for a GEM artifact detail page."""
    title_el = root.find("h1") or root.find("h2")
    title = clean(title_el.get_text(" ", strip=True)) if title_el else ""
    lines = clean_lines(root.get_text("\n", strip=True).splitlines())

    if not title and lines:
        title = lines[0]

    labels = {
        "رقم الأثر", "المجموعة", "الحِقْبةُ", "الأسرة", "الوصف",
        "المَنشأ", "الإقليم", "المنطقة", "مادة الصنع", "الأبعاد",
        "الارتفاع", "العرض", "الطول", "الوزن", "القُطر",
    }

    parts = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line in labels and i + 1 < len(lines):
            value = lines[i + 1]
            if value not in labels:
                parts.append(f"{line}: {value}")
                i += 2
                continue
        if line != title:
            parts.append(line)
        i += 1

    text = f"قطعة أثرية: {title}. " + " ".join(parts)
    text = clean(text)
    if len(text.split()) < MIN_WORDS or is_noise(text):
        return []
    return split_long(text, max_w=180, overlap=20)

# ── Driver ────────────────────────────────────────────────
def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--lang=ar")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )

# ── Per-page extraction ───────────────────────────────────
def extract_page(driver, url):
    """
    Returns (list_of_chunks, list_of_links).
    Chunks = one per heading+paragraph group (section-based).
    """
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(PAGE_WAIT)

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # ── Strip all noise containers ────────────────────
        for tag in soup(["script","style","nav","footer","header",
                          "form","iframe","noscript","aside","button","svg"]):
            tag.decompose()

        # Remove by noisy CSS classes
        noisy_cls = re.compile(
            r"cookie|privacy|modal|popup|overlay|banner|alert|toast|"
            r"social|share|newsletter|subscribe|breadcrumb|sidebar|"
            r"search|accordion-head|tab-nav|swiper|carousel|slider",
            re.I
        )
        for el in soup.find_all(True, {"class": noisy_cls}):
            el.decompose()

        # ── Find real content root ────────────────────────
        root = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id=re.compile(r"content|main", re.I)) or
            soup.find("body")
        )

        if "/collection/artefacts/" in urlparse(url).path:
            artifact_chunks = extract_artifact_chunk(root, url)
            return artifact_chunks, []

        # ── Section-based extraction ──────────────────────
        # Group: heading + all following paragraphs/li until next heading
        chunks = []
        current_heading = ""
        current_paras   = []

        def flush():
            """Turn current heading + paras into a chunk."""
            if not current_paras:
                return
            text = (current_heading + " " if current_heading else "") + \
                   " ".join(current_paras)
            text = clean(text)
            if len(text.split()) < MIN_WORDS:
                return
            if is_noise(text):
                return
            for part in split_long(text):
                if len(part.split()) >= MIN_WORDS and not is_noise(part):
                    chunks.append(part)

        HEADING_TAGS = {"h1","h2","h3","h4","h5"}
        PARA_TAGS    = {"p","li","td","dd","span"}

        for el in root.find_all(HEADING_TAGS | PARA_TAGS):
            tag  = el.name
            text = clean(el.get_text(separator=" ", strip=True))

            if not text or len(ARABIC.findall(text)) < 4:
                continue
            if is_noise(text):
                continue

            if tag in HEADING_TAGS:
                flush()
                current_heading = text
                current_paras   = []
            else:
                # Skip duplicate lines already in heading
                if text != current_heading:
                    current_paras.append(text)

        flush()   # last section

        # ── Collect Arabic links ──────────────────────────
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            full = urljoin(url, href)
            p    = urlparse(full)
            norm = p._replace(fragment="").geturl()
            if (p.netloc == urlparse(BASE_URL).netloc
                    and ("/ar/" in norm or norm.endswith("/ar"))
                    and not any(x in norm for x in IGNORE_URL)):
                links.append(norm)

        return chunks, links

    except Exception as e:
        print(f"    ⚠️  {e}")
        return [], []

# ── Crawl ─────────────────────────────────────────────────
def crawl():
    driver = make_driver()
    print("✅ Chrome ready\n")
    visited, queue = set(), list(SEED_URLS)
    all_chunks = []

    try:
        while queue and len(visited) < MAX_PAGES:
            url = queue.pop(0).rstrip("/") + "/"
            if url in visited:
                continue
            visited.add(url)

            print(f"[{len(visited)}/{MAX_PAGES}] {url}")
            page_chunks, links = extract_page(driver, url)

            if not page_chunks:
                print("    → no content")
                continue

            all_chunks.extend(page_chunks)
            print(f"    → {len(page_chunks)} sections extracted")
            for c in page_chunks[:2]:
                print(f"       · {c[:70]}…")

            for link in links:
                norm = link.rstrip("/") + "/"
                if norm not in visited and norm not in queue:
                    queue.append(norm)
    finally:
        driver.quit()
        print("\n✅ Browser closed")

    return dedup(all_chunks)

# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  GEM Arabic Scraper v3 — Section-Based")
    print(f"  Target : {BASE_URL}")
    print("="*55 + "\n")

    chunks = crawl()

    if not chunks:
        print("❌ No chunks produced.")
    else:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"\n✅ {len(chunks)} clean chunks → {OUTPUT}")
        print("\nSample chunks:")
        for c in chunks[:5]:
            print(f"  [{len(c.split())}w] {c[:80]}…\n")
        print("Next: python build_index.py")
