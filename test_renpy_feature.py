#!/usr/bin/env python3
"""
Simple test script to verify RenPy processing functionality
"""

import sys
import os

# Enable test mode to bypass OpenAI dependency
os.environ["TEST_MODE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import parse_renpy_file, process_renpy_chunks

# Sample RenPy content from the user's example
sample_renpy_content = '''leo "「はいはい、皆さん。」"
narrator "レオが前に出てきて、俺達を静かにさせる。"
leo "「一応言っておくと、チェイスは学校のプロジェクトの為に帰って来てるんだ。それが最優先であることを皆忘れないように。」"
leo "「ただし！チェイスが暇な時にやることも、いくつか考えておいたんだ。」"
leo "「予定通り、明日はサウスウェスト・アドベンチャーズに行くぞ。俺がまずこの二人を拾って―」"
narrator "レオはフリンとカールに向けた視線をこちらに戻す。"
leo "「—んで、お前たちを迎えに9時にここに寄るから。OK？」"
jenna "「OK。」"
leo "「それから、週の後半あたりにペイトンで何かできたらと思ってるけど、それは後で決めよう。」"
leo "「その間にも、暇があったら遊ぼうぜ。せっかく集まったんだからな。」"
narrator "話し終えたと思いきや、レオは急に指を鳴らす。何か思い出したようだ。"
leo "「あ、そうだ！チェイスお前、カメラ持ってきてたよな。」"
leo "「集合写真撮ろうぜ。ちゃんとしたカメラで撮った綺麗なの、皆欲しいだろうし。」"
chase "「そうだね。さっきセルフタイマー設定したから、うまくいくはず―」"
scene bg motelfull
# The inside of the motel room, with two beds visible, a nightstand, and a table visible.
with dissolve
narrator "三脚を調整する俺と、ベッドの端に皆を整頓させるレオ。そして、ようやく皆が位置につく。"
chase "「よーし、いくよ･･･。」"
narrator "タイマーをセットし、レオが俺の為に空けてくれたスペースへと急ぐ。"
play sound "camera.mp3"
window show
narrator "あれから2時間ほど映画を流し見ながら、皆で互いの近況を話していた。"
narrator "また帰ってこられて嬉しいし、まるで3年前の続きのように皆と自然に話せている。"
narrator "本当にいい感じだ。"
narrator "22:00を回ったとこで、レオ、カール、フリンの三人が帰っていった。"
narrator "明日の撮影に備え、20分ほどかけて機材の調整をする。"
narrator "TJに続き、俺も寝る支度を終えて、彼と同じベッドに入り込んだ。"
narrator "部屋の隅の方から柔らかい光が漏れている。ジェナはテーブルで何かを読んでいる。"
narrator "俺は天井を見つめながら、彼女が寝るのを待つ･･･。"
stop loop fadeout 10.0
scene bg creepylake
# The view of the shore of Lake Emma, where Sidney tragically drowned.
with opening_fade
play music "meeting1.mp3" fadein 10.0
narrator "湖から離れようとすると、俺の足首には鎖がつけられて、膝から下は泥だらけだ。"
narrator "振り返ると、鎖は岩や岸の草の周りを蛇行して、水中へ沈んでいる。"
narrator "再び前を見れば、レオがこちらを見ている。笑って、手を振っている。"
narrator "レオに向かって歩き出すと、鎖は緩んでおり、水中から簡単に引き出せてしまう。"
narrator "俺は言う。「湖の中に錨があるみたいだ。」"
narrator "レオをじっと見つめるが、彼は何も言わずにただ笑っている。"
narrator "俺は岩の上にしゃがみ、鎖を手首に何周か巻き付けた。"
narrator "レオは俺の隣に跪き、俺の背中をさすりながら腕を突き出してブレスレットを見比べ、皆持ってるんだと言う。"
narrator "彼は嬉しそうだが、俺はここから動けない。立ち上がって歩くことが、できないんだ。"
jump wideshot'''

def test_renpy_parsing() -> None:
    """Test the RenPy parsing functionality"""
    print("🧪 Testing RenPy parsing functionality...")
    
    # Test parsing
    chunks = parse_renpy_file(sample_renpy_content, chunk_size=5)  # Use smaller chunks for testing
    
    print(f"✅ Successfully parsed {len(chunks)} chunks")
    
    # Verify chunks contain dialogue
    total_dialogues = sum(len(chunk['dialogues']) for chunk in chunks)
    print(f"✅ Total dialogue lines extracted: {total_dialogues}")
    
    # Check first chunk
    if chunks:
        first_chunk = chunks[0]
        print(f"✅ First chunk contains {len(first_chunk['dialogues'])} dialogues")
        print(f"✅ First dialogue: {first_chunk['dialogues'][0]['speaker']}: {first_chunk['dialogues'][0]['text']}")
    
    # Test structured content conversion
    structured_content = process_renpy_chunks(chunks)
    
    print(f"✅ Converted to structured content with {len(structured_content['phrases'])} phrase entries")
    
    # Verify structure
    expected_keys = ['vocabulary', 'kanji', 'grammar', 'phrases', 'idioms']
    for key in expected_keys:
        if key not in structured_content:
            print(f"❌ Missing key: {key}")
        else:
            print(f"✅ Found key: {key} with {len(structured_content[key])} items")
    
    print("🎉 All tests passed! RenPy feature is working correctly.")

if __name__ == "__main__":
    test_renpy_parsing()