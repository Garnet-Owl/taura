"""
Test script for the Bible parser.

This script tests the Bible parser on sample texts to ensure it correctly
parses and aligns verses between Kikuyu and English.
"""

from .bible_parser import preprocess_bible_text, parse_kikuyu_verse, parse_english_verse, align_verses, create_dataset

# Sample Kikuyu text
KIKUYU_SAMPLE = """
Kĩambĩrĩria 1:1
1
KĨAMBĨRĨRIA
Kĩambĩrĩria 1:12
1 Kĩambĩrĩria-inĩ kĩa maũndũ mothe, Ngai nĩombire
igũrũ na thĩ. 2Hĩndĩ ĩyo thĩ ndĩarĩ ũrĩa yatariĩ na ndĩarĩ
nakĩndũ, nayo ndumayarĩigũrũrĩa kũrĩa kũriku, nake
Roho waNgaiaareerete igũrũ rĩa maaĩ.
3 Nake Ngai agĩathana, akiuga atĩrĩ, "Nĩkũgĩe ũtheri,"
na gũkĩgĩa ũtheri. 4Ngai akĩona atĩ ũtheri ũcio warĩ
mwega, nake akĩgayũkania ũtheri na nduma. 5Ngai
agĩĩta ũtheri ũcio "mũthenya," nayo ndumaakĩmĩĩta
"ũtukũ." Na gũkĩgĩa hwaĩ-inĩ, na gũkĩgĩa rũciinĩ.
Ũcio ũgĩtuĩka mũthenya wa mbere.
"""

# Sample English text
ENGLISH_SAMPLE = """
Genesis 1:1
1
Genesis 1:30
Genesis
The Creation of the World
1 In the beginning God created the heavens and the earth.
2 Now the earth was without shape and empty, and darkness was over the surface of the watery
deep, but the Spirit of God was moving over the surface of the water. 3God said, "Let there be light."
Andtherewaslight! 4Godsawthatthelightwasgood, soGodseparatedthelightfromthedarkness.
5 God called the light "day" and the darkness "night." There was evening, and there was morning,
marking the first day.
"""


def main():
    """
    Run tests on the Bible parser with sample texts.
    """
    # Test preprocessing
    print("Testing text preprocessing...")
    processed_kikuyu = preprocess_bible_text(KIKUYU_SAMPLE, "kikuyu")
    processed_english = preprocess_bible_text(ENGLISH_SAMPLE, "english")

    print("\nPreprocessed Kikuyu:")
    print(processed_kikuyu[:300] + "...")

    print("\nPreprocessed English:")
    print(processed_english[:300] + "...")

    # Test parsing
    print("\nParsing Kikuyu text...")
    kikuyu_dict = parse_kikuyu_verse(KIKUYU_SAMPLE)
    print(f"Found {sum(len(chapters) for chapters in kikuyu_dict.values())} Kikuyu chapters")

    print("\nParsing English text...")
    english_dict = parse_english_verse(ENGLISH_SAMPLE)
    print(f"Found {sum(len(chapters) for chapters in english_dict.values())} English chapters")

    # Test verse extraction
    if "Kĩambĩrĩria" in kikuyu_dict and 1 in kikuyu_dict["Kĩambĩrĩria"]:
        print("\nKikuyu verses in chapter 1:")
        for verse_num, verse_text in kikuyu_dict["Kĩambĩrĩria"][1].items():
            print(f"  Verse {verse_num}: {verse_text[:50]}...")

    if "Genesis" in english_dict and 1 in english_dict["Genesis"]:
        print("\nEnglish verses in chapter 1:")
        for verse_num, verse_text in english_dict["Genesis"][1].items():
            print(f"  Verse {verse_num}: {verse_text[:50]}...")

    # Test alignment
    print("\nAligning verses...")
    aligned_verses = align_verses(kikuyu_dict, english_dict)
    print(f"Found {len(aligned_verses)} aligned verse pairs")

    # Create dataset
    if aligned_verses:
        df = create_dataset(aligned_verses)
        print("\nDataset preview:")
        print(df.head())

    print("\nTests completed.")


if __name__ == "__main__":
    main()
