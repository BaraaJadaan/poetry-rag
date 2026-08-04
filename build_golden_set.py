"""
Builds the eval_set.json golden dataset for Phase 6.

We define 20 famous Arabic verses and the "semantic query" a user might
type to find them. This script uses FTS to locate the exact verse ID in
LanceDB, and pairs it with the semantic query to create the ground truth.
"""

import json
import sys

try:
    import lancedb
except ImportError:
    print("ERROR: lancedb not installed.")
    sys.exit(1)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

db = lancedb.connect("./lancedb")
try:
    tbl = db.open_table("ashaar_baits")
except Exception:
    print("ERROR: Could not open LanceDB table 'ashaar_baits'.")
    sys.exit(1)

# Format: (Exact text to search in DB, Semantic Query the user might type)
RAW_TARGETS = [
    ("الخيل والليل والبيداء تعرفني", "الفروسية والشجاعة وركوب الخيل في الليل"),
    ("السيف أصدق أنباء من الكتب", "صدق السيف في الحرب أفضل من الكتب"),
    ("الأم مدرسة إذا أعددتها", "تربية الأم الصالحة تبني الأجيال"),
    ("ألا ليت الشباب يعود يوما", "ليت مرحلة الشباب تعود يوما لأخبرها"),
    ("قف نبك من ذكرى حبيب ومنزل", "البكاء عند الوقوف على منزل الحبيب"),
    ("إذا الشعب يوما أراد الحياة", "إذا أراد الشعب الحياة فلا بد أن ينتصر"),
    ("وما نيل المطالب بالتمني", "تحقيق المطالب لا يأتي بالتمني بل بالجهد"),
    ("أنا الذي نظر الأعمى إلى أدبي", "فخر الشاعر بأن الأعمى يرى أدبه وشعره"),
    ("نعيب زماننا والعيب فينا", "نعيب الزمان والزمن بريء والعيب فينا"),
    ("هل غادر الشعراء من متردم", "هل ترك الشعراء شيئا جديدا لنقوله"),
    ("تعيرنا أنا قليل عديدنا", "الرد على من يعيرنا بقلة العدد"),
    ("دقات قلب المرء قائلة له", "دقات القلب تذكرنا بأن الحياة قصيرة"),
    ("على قدر أهل العزم تأتي العزائم", "تأتي العزائم والإنجازات على قدر أهل العزم"),
    ("ولا خير في ود امرئ متلون", "لا خير في ود صديق متلون يتغير بسرعة"),
    ("بانت سعاد فقلبي اليوم متبول", "فراق سعاد ترك قلبي حزينا"),
    ("لا تسقني ماء الحياة بذلة", "أفضل الموت على شرب ماء الحياة بذلة"),
    ("ومن يتهيب صعود الجبال", "الذي يخاف صعود الجبال يعيش في الحفر"),
    ("إذا رأيت نيوب الليث بارزة", "لا تظن أن الليث يبتسم إذا برزت أنيابه"),
    ("فيا ليت الذي بيني وبينك عامر", "ليت ما بيني وبينك عامر حتى لو خرب الناس"),
    ("قم للمعلم وفه التبجيلا", "قم احتراما للمعلم وفه التبجيلا"),
]

print("Building Golden Set (eval_set.json)...")
golden_set = []

for exact_text, semantic_query in RAW_TARGETS:
    # Use FTS to find the exact verse in our 3.4M rows
    res = tbl.search(exact_text, query_type="fts").limit(1).to_pandas()
    
    if res.empty:
        print(f"[WARN] Could not find verse in DB: '{exact_text}'")
        continue
        
    row = res.iloc[0]
    golden_set.append({
        "query": semantic_query,
        "expected_id": row["id"],
        "expected_text": row.get("text_display") or row.get("text_index"),
        "poet": row.get("poet_name", "Unknown")
    })

print(f"Successfully matched {len(golden_set)}/{len(RAW_TARGETS)} verses.")

with open("eval_set.json", "w", encoding="utf-8") as f:
    json.dump(golden_set, f, ensure_ascii=False, indent=2)

print("Saved eval_set.json")
