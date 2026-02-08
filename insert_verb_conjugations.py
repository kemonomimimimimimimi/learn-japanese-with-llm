#!/usr/bin/env python3
"""
One-time script to bulk insert all Japanese verb conjugation rows
into the database.
"""

from llm_learn_japanese import db
from llm_learn_japanese.db import VerbConjugation, get_session, add_conjugation

# Master checklist of verb conjugations
CONJUGATIONS = [
    ("Core verb forms", "Dictionary / plain (non-past)"),
    ("Core verb forms", "ます-stem（連用形）"),
    ("Core verb forms", "ない-stem（未然形）"),
    ("Core verb forms", "て-form（テ形）"),
    ("Core verb forms", "た-form（タ形 / past, completion）"),
    ("Core verb forms", "Volitional（推量形：～う／～よう）"),
    ("Core verb forms", "Imperative（命令形：五段＝～え、 一段＝～ろ／～よ）"),
    ("Core verb forms", "Conditional: ～ば（仮定形）"),
    ("Core verb forms", "Conditional: ～たら（“when/if” after completion）"),
    ("Core verb forms", "Conditional: ～と（確定条件）"),
    ("Polite paradigm", "～ます／～ません（非過去）"),
    ("Polite paradigm", "～ました／～ませんでした（過去）"),
    ("Polite paradigm", "～ましょう（意向）"),
    ("Polite paradigm", "～まして（連用・接続）"),
    ("Polite paradigm", "～ますれば（条件）"),
    ("Polite paradigm", "～なさい（準命令）／～な（禁止・俗）"),
    ("Negative paradigm", "～ない（否定）"),
    ("Negative paradigm", "～なかった（否定過去）"),
    ("Negative paradigm", "～なくて（否定テ形）"),
    ("Negative paradigm", "～ないで（without ～ing）"),
    ("Negative paradigm", "～なければ／～なきゃ（条件）"),
    ("Negative paradigm", "～なかろう（否定意向・文語）"),
    ("Negative paradigm", "～ず（文語否定）／～ずに（文語「～ないで」）"),
    ("Voice / ability / causation", "Potential（可能）"),
    ("Voice / ability / causation", "Passive（受け身・尊敬）"),
    ("Voice / ability / causation", "Causative（使役）"),
    ("Voice / ability / causation", "Causative-passive（使役受け身）"),
    ("Te-form combos", "～ている（進行・習慣・結果状態）"),
    ("Te-form combos", "～てある（結果の残存）"),
    ("Te-form combos", "～ておく（準備）"),
    ("Te-form combos", "～ていく／～てくる（状態変化の方向性）"),
    ("Te-form combos", "～てしまう（完了・遺憾）"),
    ("Te-form combos", "～てみる（試み）"),
    ("Te-form combos", "～てほしい（依頼・希望）"),
    ("Te-form combos", "～てください（依頼）"),
    ("Te-form combos", "～ては いけない（禁止）"),
    ("Te-form combos", "～ても いい（許可）"),
    ("Te-form combos", "～た ほうが いい（助言）"),
    ("Desire / tendency", "～たい（願望）"),
    ("Desire / tendency", "～たがる（第三者の願望）"),
    ("Desire / tendency", "～がち（傾向）"),
    ("Desire / tendency", "～ながら（同時進行）"),
    ("Desire / tendency", "～方（方法）"),
    ("Desire / tendency", "複合動詞（～始める／～続ける など）"),
    ("Hearsay / supposition", "～そうだ（様態）"),
    ("Hearsay / supposition", "～そうだ（伝聞）"),
    ("Hearsay / supposition", "～らしい推量"),
    ("Hearsay / supposition", "～みたいだ"),
    ("Hearsay / supposition", "～はずだ"),
    ("Hearsay / supposition", "～つもりだ"),
    ("Hearsay / supposition", "～べきだ"),
    ("Hearsay / supposition", "～まい"),
    ("Hearsay / supposition", "～だろう／～でしょう"),
    ("Nominalization", "～の（名詞化／強調）"),
    ("Nominalization", "～こと（名詞化）"),
    ("Nominalization", "ことができる（能力表現）"),
    ("Nominalization", "ことがある（経験）"),
    ("Nominalization", "ことにする／ことになる"),
    ("Obligation / necessity", "～ないと（いけない／だめ）"),
    ("Obligation / necessity", "～なくては（いけない）"),
    ("Obligation / necessity", "～なければ（いけない）"),
    ("Obligation / necessity", "～しなくては／～しては いけない"),
    ("授受表現", "（～て）あげる／くれる／もらう"),
    ("授受表現", "すみません＋～て"),
    ("五段・一段ポイント", "五段：語尾別の活用列"),
    ("五段・一段ポイント", "一段：～いる／～える"),
    ("五段・一段ポイント", "て／た への音便規則"),
    ("不規則動詞", "する（全活用）"),
    ("不規則動詞", "来る（全活用）"),
    ("不規則動詞", "名詞＋する（サ変複合）"),
    ("敬語", "尊敬：お＋連用形＋になる／～れる"),
    ("敬語", "謙譲：お／ご＋連用形＋する"),
    ("敬語", "不規則敬語動詞"),
    ("形容詞・形容動詞", "い形容詞全活用"),
    ("形容詞・形容動詞", "な形容詞全活用"),
    ("形容詞・形容動詞", "いい→よい 注意"),
    ("コピュラ", "だ／です 系列")
]

def main() -> None:
    session = get_session()
    inserted, skipped = 0, 0
    for category, label in CONJUGATIONS:
        existing = session.query(VerbConjugation).filter_by(label=label).first()
        if existing:
            skipped += 1
            continue
        # Use helper to auto-generate CardAspects + scheduling
        if add_conjugation(label, category):
            inserted += 1
        else:
            skipped += 1
    print(f"✅ Inserted {inserted} conjugations, skipped {skipped} duplicates.")

if __name__ == "__main__":
    main()