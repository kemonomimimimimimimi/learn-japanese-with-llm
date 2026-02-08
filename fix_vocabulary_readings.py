#!/usr/bin/env python3
"""
Script to fix vocabulary entries in the database that are missing readings and meanings
by using the LLM to fill in the missing information.
"""

import sys
import os
import json
from typing import List, Tuple

# Check for OpenAI API key before proceeding
if "OPENAI_API_KEY" not in os.environ:
    print("Error: OPENAI_API_KEY environment variable is required")
    print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library is required")
    print("Please install it with: pip install openai")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_learn_japanese import db

def get_vocabulary_info_from_llm(words: List[str]) -> dict[str, dict[str, str]]:
    """Use LLM to get readings and meanings for vocabulary words."""
    
    words_text = ", ".join(words[:20])  # Process in batches of 20
    
    prompt = f"""
You are a Japanese language expert. For each Japanese word provided, give the reading (in hiragana/katakana) and English meaning.

Words: {words_text}

Return ONLY a JSON object with this exact format:
{{
  "word1": {{"reading": "hiragana/katakana", "meaning": "English meaning"}},
  "word2": {{"reading": "hiragana/katakana", "meaning": "English meaning"}},
  ...
}}

Examples:
- 勇気: {{"reading": "ゆうき", "meaning": "courage, bravery"}}
- 物語: {{"reading": "ものがたり", "meaning": "story, tale"}}
- 人生: {{"reading": "じんせい", "meaning": "life, human life"}}
- フィクション: {{"reading": "フィクション", "meaning": "fiction"}}

Be accurate and provide common readings and meanings. Return ONLY the JSON, no explanations.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a Japanese language expert. Provide accurate readings and meanings for Japanese vocabulary."},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_completion_tokens=16384,
        )
        
        # Fix: use dict access with better debugging
        msg = response.choices[0].message

        # Some SDK variants use dict, some use objects
        raw_content = None
        if isinstance(msg, dict):
            raw_content = msg.get("content")
        else:
            try:
                raw_content = getattr(msg, "content", None)
            except Exception:
                raw_content = None

        if not raw_content or not raw_content.strip():
            print("⚠️ Empty response content from LLM")
            print(f"   Full raw response object: {response}")
            return {}
        
        print(f"📤 LLM raw response snippet: {raw_content[:200]}...")
        
        try:
            vocab_info: dict[str, dict[str, str]] = json.loads(raw_content)
            return vocab_info
        except Exception as parse_err:
            print(f"❌ JSON parse failed: {parse_err}")
            print(f"   Raw content was: {raw_content[:500]}")
            return {}

    except Exception as e:
        print(f"❌ LLM request failed: {e}")
        return {}

def should_have_reading(word: str) -> bool:
    """Determine if a word should have a reading added."""
    import re

    # Skip if it's clearly a sound effect (repeated kana, sound patterns)
    if re.search(r'[ッー〜]{2,}', word):
        return False

    # Skip single katakana sound effects
    if re.match(r'^[ァ-ヶ]+$', word) and len(word) <= 4 and any(char in word for char in 'ッグキュブォザジュピ'):
        return False

    # Skip if it's all katakana and looks like a foreign name or error
    if re.match(r'^[ァ-ヶ・]+$', word):
        # Common Japanese katakana words we should keep
        japanese_katakana = [
            'フィクション', 'アルバイト', 'コンビニ', 'ショック', 'アニメ', 'ゲーム',
            'ママ', 'パパ', 'テレビ', 'ラジオ', 'カメラ', 'コンピューター', 'メール',
            'ニュース', 'スポーツ', 'レストラン', 'ホテル', 'タクシー', 'バス'
        ]

        # Skip if longer than 3 characters and not in our whitelist (likely foreign name)
        if len(word) > 3 and word not in japanese_katakana:
            return False

        # Skip very short katakana that look like typos/errors
        if len(word) <= 2 and word not in ['ママ', 'パパ']:
            return False

        # Skip if it contains unusual katakana combinations that suggest OCR errors
        if any(bad_combo in word for bad_combo in ['マパ', 'ナマ', 'ハマ', 'バマ', 'ダマ']):
            return False

    # Skip if it's just a number
    if re.match(r'^\d+$', word):
        return False

    # Skip if it's just punctuation or very short exclamations
    if len(word) <= 2 and re.match(r'^[！？!?よしあはい]+$', word):
        return False

    # Skip obvious sound effects by pattern
    sound_effect_patterns = ['キュ', 'グイ', 'ブォ', 'ガー', 'ドン', 'バン', 'ピュ', 'ザー', 'ゴー']
    if any(pattern in word for pattern in sound_effect_patterns):
        return False

    # Skip if it contains full-width spaces or looks like a full name
    if '　' in word or '・' in word:
        return False

    # Skip punctuation-only entries
    if re.match(r'^[…、。！？!?]+$', word):
        return False

    # Skip if mostly punctuation
    if len(word) <= 3 and re.search(r'[…、。！？!?]', word):
        return False

    return True

def fix_vocabulary_readings() -> None:
    """Fix vocabulary entries missing readings and meanings."""
    print("🔧 Fixing Vocabulary Readings and Meanings")
    print("=" * 50)
    
    session = db.get_session()
    
    try:
        # Find vocabulary items missing readings or meanings
        incomplete_vocab = session.query(db.Vocabulary).filter(
            (db.Vocabulary.reading == None) |
            (db.Vocabulary.reading == "") |
            (db.Vocabulary.meaning == None) |
            (db.Vocabulary.meaning == "")
        ).all()
        
        print(f"🔍 Found {len(incomplete_vocab)} vocabulary items needing fixes")
        
        # Filter out items that shouldn't have readings
        filterable_vocab = []
        skipped_items = []

        for item in incomplete_vocab:
            if should_have_reading(item.word):
                filterable_vocab.append(item)
            else:
                skipped_items.append(item)
                print(f"⏭️  Skipping '{item.word}' (sound effect/name/number)")

        print(f"📝 Will process {len(filterable_vocab)} items (skipped {len(skipped_items)} sound effects/names)")

        if not filterable_vocab:
            print("✅ All processable vocabulary items already have readings and meanings!")
            return
        
        # Process in batches to avoid overwhelming the API
        batch_size = 15
        total_fixed = 0
        
        for i in range(0, len(filterable_vocab), batch_size):
            batch = filterable_vocab[i:i + batch_size]
            words = [item.word for item in batch]
            
            print(f"\n🤖 Processing batch {i//batch_size + 1}/{(len(filterable_vocab) + batch_size - 1)//batch_size}")
            print(f"   Words: {', '.join(words[:5])}{'...' if len(words) > 5 else ''}")
            print(f"   🔍 Full word list being sent: {words}")
            
            # Get info from LLM
            vocab_info = get_vocabulary_info_from_llm(words)
            
            if not vocab_info:
                print(f"⚠️  Skipping batch due to LLM error (might be sound effects or names)")
                continue
            
            # Update database entries
            for item in batch:
                word_info = vocab_info.get(item.word, {})
                
                if word_info:
                    old_reading = item.reading or "N/A"
                    old_meaning = item.meaning or "N/A"
                    
                    # Update reading if missing
                    if not item.reading or item.reading.strip() == "":
                        new_reading = word_info.get("reading", "")
                        if new_reading:
                            item.reading = new_reading
                            print(f"   📚 {item.word}: Added reading '{new_reading}'")
                    
                    # Update meaning if missing  
                    if not item.meaning or item.meaning.strip() == "":
                        new_meaning = word_info.get("meaning", "")
                        if new_meaning:
                            item.meaning = new_meaning
                            print(f"   💡 {item.word}: Added meaning '{new_meaning}'")
                    
                    total_fixed += 1
                else:
                    print(f"   ⚠️  No info returned for: {item.word}")
        
        # Commit all changes
        session.commit()
        print(f"\n✅ Fixed {total_fixed} vocabulary entries!")
        
        # Show some examples of what was fixed
        print(f"\n📋 Sample of fixed entries:")
        sample_fixed = session.query(db.Vocabulary).filter(
            (db.Vocabulary.reading != None) & 
            (db.Vocabulary.reading != "") &
            (db.Vocabulary.meaning != None) &
            (db.Vocabulary.meaning != "")
        ).limit(5).all()
        
        for item in sample_fixed:
            print(f"   {item.word} | {item.reading} | {item.meaning}")
            
    except Exception as e:
        print(f"❌ Error fixing vocabulary: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    # Check if database exists
    if not os.path.exists("japanese_learning.db"):
        print("❌ Database file 'japanese_learning.db' not found!")
        print("   Make sure you're running this from the correct directory.")
        sys.exit(1)
    
    fix_vocabulary_readings()
    print(f"\n🎉 Vocabulary fixing complete! Run 'python check_database.py' to see the results.")