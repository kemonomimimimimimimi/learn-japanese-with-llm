#!/usr/bin/env python3
"""
Script to examine the contents of the Japanese learning database
to see what was actually extracted and stored.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_learn_japanese import db

def check_database_contents() -> None:
    """Examine the database contents to see what was extracted."""
    print("🔍 Examining Japanese Learning Database Contents")
    print("=" * 60)
    
    session = db.get_session()
    
    try:
        # Check vocabulary
        vocab_items: list[db.Vocabulary] = session.query(db.Vocabulary).all()
        print(f"\n📚 VOCABULARY ({len(vocab_items)} items):")
        for i, vocab in enumerate(vocab_items[-10:], 1):  # Show last 10
            print(f"  {i:2d}. {vocab.word} | Reading: {vocab.reading or 'N/A'} | Meaning: {vocab.meaning or 'N/A'}")
        if len(vocab_items) > 10:
            print(f"     ... and {len(vocab_items) - 10} more items")
        
        # Check kanji
        kanji_items: list[db.Kanji] = session.query(db.Kanji).all()
        print(f"\n🈷️  KANJI ({len(kanji_items)} items):")
        for i, kanji in enumerate(kanji_items[-10:], 1):  # Show last 10
            print(f"  {i:2d}. {kanji.character} | Onyomi: {kanji.onyomi or 'N/A'} | Kunyomi: {kanji.kunyomi or 'N/A'} | Meanings: {kanji.meanings or 'N/A'}")
        if len(kanji_items) > 10:
            print(f"     ... and {len(kanji_items) - 10} more items")
        
        # Check grammar
        grammar_items: list[db.Grammar] = session.query(db.Grammar).all()
        print(f"\n📝 GRAMMAR ({len(grammar_items)} items):")
        for i, grammar in enumerate(grammar_items[-10:], 1):  # Show last 10
            print(f"  {i:2d}. {grammar.point} | Explanation: {grammar.explanation or 'N/A'}")
        if len(grammar_items) > 10:
            print(f"     ... and {len(grammar_items) - 10} more items")
        
        # Check phrases
        phrase_items: list[db.Phrase] = session.query(db.Phrase).all()
        print(f"\n💬 PHRASES ({len(phrase_items)} items):")
        for i, phrase in enumerate(phrase_items[-10:], 1):  # Show last 10
            print(f"  {i:2d}. {phrase.phrase} | Meaning: {phrase.meaning or 'N/A'}")
        if len(phrase_items) > 10:
            print(f"     ... and {len(phrase_items) - 10} more items")
        
        # Check idioms
        idiom_items: list[db.Idiom] = session.query(db.Idiom).all()
        print(f"\n🎭 IDIOMS ({len(idiom_items)} items):")
        for i, idiom in enumerate(idiom_items[-10:], 1):  # Show last 10
            print(f"  {i:2d}. {idiom.idiom} | Meaning: {idiom.meaning or 'N/A'}")
        if len(idiom_items) > 10:
            print(f"     ... and {len(idiom_items) - 10} more items")
        
        # Check card aspects
        aspect_items = session.query(db.CardAspect).all()
        print(f"\n🃏 CARD ASPECTS ({len(aspect_items)} items):")
        print("     (These are the individual learning cards generated from the content)")
        
        # Summary
        total_content = len(vocab_items) + len(kanji_items) + len(grammar_items) + len(phrase_items) + len(idiom_items)
        print(f"\n📊 SUMMARY:")
        print(f"     Total Content Items: {total_content}")
        print(f"     Total Learning Cards: {len(aspect_items)}")
        
        # Show most recent additions (by ID)
        print(f"\n🕒 MOST RECENTLY ADDED CONTENT:")
        
        recent_vocab = session.query(db.Vocabulary).order_by(db.Vocabulary.id.desc()).limit(5).all()
        if recent_vocab:
            print("   Recent Vocabulary:")
            for vocab in recent_vocab:
                print(f"     - {vocab.word} ({vocab.reading or 'no reading'})")
        
        recent_kanji: list[db.Kanji] = session.query(db.Kanji).order_by(db.Kanji.id.desc()).limit(5).all()
        if recent_kanji:
            print("   Recent Kanji:")
            for item in recent_kanji:
                print(f"     - {item.character} ({item.meanings or 'no meaning'})")
        
    except Exception as e:
        print(f"❌ Error examining database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    # Check if database exists
    if not os.path.exists("japanese_learning.db"):
        print("❌ Database file 'japanese_learning.db' not found!")
        print("   Make sure you're running this from the correct directory.")
        sys.exit(1)
    
    check_database_contents()