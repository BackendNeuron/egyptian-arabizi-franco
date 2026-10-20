"""Sentence patterns with interchangeable parts. Every part is written in both scripts, so each generated
sentence comes with a real human-style spelling (then varied by arabizi.variants like the seeds).
Patterns and parts are kept grammatical together: subjects carry the verb form they need."""

import itertools
import random

# (arabic, arabizi) pairs
PEOPLE_GOING = [  # subject + "going" agreeing with it
    (("انا رايح", "ana raye7"), ("انا رايحة", "ana ray7a")), (("احنا رايحين", "e7na ray7een"),),
    (("هو رايح", "howa raye7"),), (("هي رايحة", "heya ray7a"),), (("انت رايح", "enta raye7"),),
    (("انتو رايحين", "ento ray7een"),), (("صاحبي رايح", "sa7by raye7"),), (("ماما رايحة", "mama ray7a"),),
]
PLACES = [("الشغل", "el shoghl"), ("الجامعة", "el gam3a"), ("البيت", "el bet"), ("النادي", "el nady"),
          ("السوق", "el soo2"), ("الدكتور", "el doktor"), ("البنك", "el bank"), ("المطار", "el matar"),
          ("الجيم", "el gym"), ("المول", "el mall"), ("القهوة", "el 2ahwa"), ("اسكندرية", "eskendereya"),
          ("الساحل", "el sa7el"), ("المدرسة", "el madrasa"), ("الكافيه", "el cafe"), ("المستشفى", "el mostashfa")]
TIMES = [("بكرة", "bokra"), ("النهاردة", "elnaharda"), ("دلوقتي", "dlwa2ty"), ("بعد شوية", "ba3d shwaya"),
         ("الصبح", "el sob7"), ("بالليل", "bel leil"), ("الساعة اتنين", "el sa3a etnen"),
         ("الساعة تمانية", "el sa3a tamanya"), ("يوم السبت", "yom el sabt"), ("الاسبوع الجاي", "el esboo3 el gay"),
         ("بعد الضهر", "ba3d el dohr"), ("كمان ساعة", "kaman sa3a")]
WANT = [("انا عايز", "ana 3ayez"), ("انا عايزة", "ana 3ayza"), ("احنا عايزين", "e7na 3ayzeen"),
        ("هو عايز", "howa 3ayez"), ("هي عايزة", "heya 3ayza"), ("انت عايز", "enta 3ayez")]
THINGS = [("شاي", "shay"), ("قهوة", "2ahwa"), ("مية", "maya"), ("عصير", "3aseer"), ("فلوس", "flous"),
          ("موبايل جديد", "mobile gedid"), ("عربية", "3arabeya"), ("اجازة", "agaza"), ("اكل", "akl")]
# verbs only after "I want", so the person agrees
I_WANT = [("انا عايز", "ana 3ayez"), ("انا عايزة", "ana 3ayza")]
MY_VERBS = [("انام", "anam"), ("اخرج", "a5rog"), ("اسافر", "asafer"), ("اتكلم معاك", "atkalem ma3ak"),
            ("اشوفك", "ashoofak"), ("ارتاح شوية", "arta7 shwaya"), ("اروح", "aro7"), ("اكل", "akol")]
QUALITIES = [("حلو", "7elw"), ("وحش", "we7esh"), ("غالي", "ghaly"), ("رخيص", "rekhees"), ("كبير", "kbeer"),
             ("صغير", "soghayar"), ("جديد", "gedid"), ("قديم", "2adeem"), ("زحمة", "za7ma"), ("بعيد", "be3eed")]
THIS = [("المكان ده", "el makan da"), ("الموبايل ده", "el mobile da"), ("المطعم ده", "el mat3am da"),
        ("الفيلم ده", "el film da"), ("الشغل ده", "el shoghl da"), ("البيت ده", "el bet da"),
        ("الكلام ده", "el kalam da"), ("اللبس ده", "el lebs da")]
DEGREE = [("اوي", "awy"), ("جدا", "gedan"), ("شوية", "shwaya"), ("خالص", "5ales"), ("", "")]


def _join(*parts):
    ar = " ".join(p[0] for p in parts if p[0])
    az = " ".join(p[1] for p in parts if p[1])
    return ar, az


def generate(rnd: random.Random | None = None):
    rnd = rnd or random.Random(0)
    out = set()
    for forms in PEOPLE_GOING:
        for s, pl, t in itertools.product(forms, PLACES, TIMES):
            out.add(_join(s, pl, t))
    for w, th in itertools.product(WANT, THINGS):
        out.add(_join(w, th))
        out.add(_join(w, th, rnd.choice(TIMES)))
    for w, v in itertools.product(I_WANT, MY_VERBS):
        out.add(_join(w, v))
        out.add(_join(w, v, rnd.choice(TIMES)))
    for th, q, d in itertools.product(THIS, QUALITIES, DEGREE):
        out.add(_join(th, q, d))
        out.add(_join(th, ("مش", "msh"), q))
    return sorted(out)


if __name__ == "__main__":
    pairs = generate()
    print(len(pairs))
    for p in random.Random(3).sample(pairs, 8):
        print(p)
