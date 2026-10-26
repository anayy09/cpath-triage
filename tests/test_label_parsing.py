"""
Regression tests for src/models/parsing.py.

Every response below is copied verbatim from a saved output, with the label the
v5.0 whole-text parser stored for it. The first set is where that parser took the
label from the reasoning sentence or from a list of classes; the second is where
both parsers must agree.
"""

from src.models.parsing import parse_prediction, parse_prediction_regex

TASK = "tissue_classification"

# results/zeroshot/medgemma-27b-it/V3_full: the model answered background.
BG_TO_DEBRIS = (
    "LABEL: background\nCONFIDENCE: 95\nREASONING: The image shows a mostly empty field with some "
    "scattered cellular debris and indistinct structures, typical of background or empty space in a "
    "histological section."
)
# results/zeroshot/gemma-3-27b-it/V3_full: the model answered stroma.
STROMA_TO_TUMOUR = (
    "LABEL: cancer-associated stroma\nCONFIDENCE: 85\nREASONING: The image shows a dense, reactive "
    "fibrous tissue with scattered cells, consistent with the desmoplastic reaction often seen "
    "surrounding colorectal adenocarcinoma."
)
# results/consistency/medgemma-27b-it/V3W_test, variant 4, cut at the 128-token cap.
CLASS_LIST = (
    "LABEL: adipose\nCONFIDENCE: 95\nREASONING: The image shows large, rounded cells with "
    "empty-appearing cytoplasm and thin membranes, characteristic of adipocytes (fat cells).\n\n"
    "LABEL: background\nCONFIDENCE: 100\nREASONING: The white, empty space surrounding the tissue "
    "elements represents the background, likely formalin fixation artifact or empty space in the "
    "tissue section.\n\nLABEL: lymphocytes\nCONFIDENCE: 85\nREASONING: Small, densely packed cells "
    "with dark, round nuclei and scant cytoplasm are visible, consistent with lymphocytes.\n\n"
    "LABEL: smooth muscle\nCONFIDENCE: 7"
)
# results/zeroshot/medgemma-27b-it/V3_full: a shortened class name.
SHORT_NAME = (
    "LABEL: adenocarcinoma epithelium\nCONFIDENCE: 95\nREASONING: The image shows irregular, crowded "
    "glands with atypical epithelial cells, characteristic of colorectal adenocarcinoma."
)
# results/zeroshot/gemma-3-27b-it/V3_full_test: an answer outside the class list.
OFF_LIST = (
    "LABEL: fungal hyphae\nCONFIDENCE: 95\nREASONING: The image shows branching, filamentous "
    "structures with distinct walls and spores, characteristic of fungal elements within the tissue."
)


def test_label_comes_from_the_label_line_not_the_reasoning():
    assert parse_prediction_regex(BG_TO_DEBRIS, TASK) == ("debris", 0.95)
    assert parse_prediction(BG_TO_DEBRIS, TASK) == ("background", 0.95)
    assert parse_prediction_regex(STROMA_TO_TUMOUR, TASK)[0] == "colorectal adenocarcinoma epithelium"
    assert parse_prediction(STROMA_TO_TUMOUR, TASK) == ("cancer-associated stroma", 0.85)


def test_a_list_of_classes_is_not_a_label():
    assert parse_prediction_regex(CLASS_LIST, TASK) == ("smooth muscle", 0.95)
    label, conf = parse_prediction(CLASS_LIST, TASK)
    assert label == "unknown"
    # The first stated confidence, as the v5.0 rule also read it.
    assert conf == 0.95


def test_shortened_name_and_off_list_answer_agree_with_v5():
    assert parse_prediction(SHORT_NAME, TASK) == parse_prediction_regex(SHORT_NAME, TASK)
    assert parse_prediction(SHORT_NAME, TASK)[0] == "colorectal adenocarcinoma epithelium"
    assert parse_prediction(OFF_LIST, TASK) == parse_prediction_regex(OFF_LIST, TASK) == ("unknown", 0.95)


def test_no_label_line_falls_back_to_the_whole_text_search():
    text = "This patch shows dense sheets of lymphocytes. Confidence: 80"
    assert parse_prediction(text, TASK) == parse_prediction_regex(text, TASK) == ("lymphocytes", 0.8)


def test_markdown_bold_and_trailing_period_are_tolerated():
    text = "**LABEL:** Mucus.\n**CONFIDENCE:** 90\nREASONING: Pale pools of extracellular material."
    assert parse_prediction(text, TASK) == ("mucus", 0.9)
