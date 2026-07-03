"""
src/models/prompts.py

Prompt templates for the colorectal H&E tissue classification task. client.py
imports get_prompt() to build the default prompt for a task. Keep the label
vocabulary here consistent with the parser in client._parse_prediction.

The nine NCT-CRC tissue classes (PathMNIST label space):
    ADI  adipose
    BACK background
    DEB  debris
    LYM  lymphocytes
    MUC  mucus
    MUS  smooth muscle
    NORM normal colon mucosa
    STR  cancer-associated stroma
    TUM  colorectal adenocarcinoma epithelium (tumor)
"""

from __future__ import annotations

TISSUE_CLASSES = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]

_CLASS_BLOCK = "\n".join(f"- {c}" for c in TISSUE_CLASSES)

# V1: minimal. Baseline to see raw behavior.
_V1 = (
    "You are a pathology assistant. The image is a hematoxylin and eosin (H&E) "
    "stained tissue patch from a colorectal sample. Classify it into exactly one "
    "of these nine tissue types:\n"
    f"{_CLASS_BLOCK}\n\n"
    "Answer with the tissue type and a confidence percentage."
)

# V2: structured output, easier to parse. Default.
_V2 = (
    "You are an expert pathology assistant analyzing a hematoxylin and eosin "
    "(H&E) stained tissue patch from a colorectal specimen.\n\n"
    "Classify the patch into exactly one of these nine tissue types:\n"
    f"{_CLASS_BLOCK}\n\n"
    "{clinical_context}\n\n"
    "Respond in exactly this format and nothing else:\n"
    "LABEL: <one tissue type from the list above>\n"
    "CONFIDENCE: <integer 0-100>\n"
    "REASONING: <one sentence naming the histological features you used>"
)

# V3: adds brief feature cues per class to anchor the model. Use for the
# few-shot / harder ablation, not as the clean baseline.
_V3 = (
    "You are an expert gastrointestinal pathologist analyzing an H&E stained "
    "colorectal tissue patch.\n\n"
    "Tissue types and their typical histological cues:\n"
    "- adipose: large empty rounded cells, thin membranes\n"
    "- background: white or near-empty glass, no tissue\n"
    "- debris: amorphous necrotic material, no intact cells\n"
    "- lymphocytes: dense small round dark nuclei, scant cytoplasm\n"
    "- mucus: pale wispy extracellular material\n"
    "- smooth muscle: elongated eosinophilic fibers, cigar-shaped nuclei\n"
    "- normal colon mucosa: regular crypts, organized epithelium\n"
    "- cancer-associated stroma: reactive fibrous tissue around tumor\n"
    "- colorectal adenocarcinoma epithelium: irregular crowded malignant glands\n\n"
    "{clinical_context}\n\n"
    "Respond in exactly this format and nothing else:\n"
    "LABEL: <one tissue type from the list above>\n"
    "CONFIDENCE: <integer 0-100>\n"
    "REASONING: <one sentence>"
)

# V4: chain-of-thought preamble + explicit disambiguation for the three zero-F1 classes.
# V3 broke NORM (was 0.61) and STR (was 0.18) by over-cuing smooth muscle. V4 corrects
# this with hard rules: STR requires glandular context, NORM requires organized crypts,
# MUC requires extracellular mucin pools as the dominant feature.
_V4 = (
    "You are an expert gastrointestinal pathologist analyzing an H&E stained "
    "colorectal tissue patch.\n\n"
    "Before choosing a label, reason through these steps. "
    "Do not write the reasoning steps in your response.\n"
    "Step 1: Identify the dominant structural feature visible in the patch "
    "(organized glands or crypts, dense lymphocyte sheets, acellular pools, "
    "or spindle-cell fibers).\n"
    "Step 2: Eliminate classes whose required features are absent.\n"
    "Step 3: Apply the disambiguation rules below to resolve remaining candidates.\n\n"
    "Tissue classes and their required diagnostic features:\n"
    "- adipose: large empty rounded vacuoles, thin membranes, nuclei absent\n"
    "- background: white or near-empty glass, no tissue present\n"
    "- debris: amorphous necrotic material, cellular ghosts, no intact cells\n"
    "- lymphocytes: dense sheets of small round dark nuclei, scant cytoplasm\n"
    "- mucus: pale acellular or sparsely cellular pools of extracellular mucin; "
    "the dominant feature is blue-gray homogeneous pools with very few intact cells; "
    "CHOOSE mucus when extracellular mucin pools cover most of the patch.\n"
    "- smooth muscle: organized parallel fascicles of uniformly elongated spindle "
    "cells with blunt cigar-shaped nuclei, abundant eosinophilic cytoplasm; "
    "NO glandular structures are present anywhere in the patch.\n"
    "- normal colon mucosa: regular symmetric crypts, columnar epithelium, goblet "
    "cells, preserved crypt architecture, no invasion; "
    "CHOOSE normal colon mucosa when well-organized crypts are present with no "
    "malignant features.\n"
    "- cancer-associated stroma: spindle-cell-rich reactive or desmoplastic fibrous "
    "tissue situated adjacent to or infiltrating malignant glands; loose irregular "
    "collagen bundles and reactive fibroblasts in a fibrous matrix; "
    "CHOOSE cancer-associated stroma only when reactive fibrous tissue is in contact "
    "with or surrounds malignant glands. Without visible glandular context, prefer "
    "smooth muscle over cancer-associated stroma.\n"
    "- colorectal adenocarcinoma epithelium: irregular crowded malignant glands, "
    "nuclear pleomorphism, loss of polarity, invasive architecture\n\n"
    "Disambiguation rules:\n"
    "Rule 1 (smooth muscle vs cancer-associated stroma): Both have spindle cells. "
    "Smooth muscle shows uniform parallel fascicles with NO glandular context. "
    "Cancer-associated stroma occurs only adjacent to malignant glands. "
    "If no glands are visible, the label is smooth muscle, not cancer-associated stroma.\n"
    "Rule 2 (normal colon mucosa vs cancer-associated stroma): Normal colon mucosa "
    "has well-organized regular crypts with columnar epithelium. "
    "Cancer-associated stroma has no organized crypts. "
    "If you see orderly crypts, the label is normal colon mucosa.\n"
    "Rule 3 (mucus vs normal colon mucosa): Mucus patches are dominated by pale "
    "extracellular pools with sparse cells and no organized crypts. "
    "Normal colon mucosa has intact glandular crypts. "
    "If the patch shows pale pooled material without intact crypts, choose mucus.\n\n"
    "{clinical_context}\n\n"
    "Respond in exactly this format and nothing else:\n"
    "LABEL: <one tissue type from the list above>\n"
    "CONFIDENCE: <integer 0-100>\n"
    "REASONING: <one sentence naming the dominant histological features you observed>"
)

# V4b: fixes TUM and ADI regressions introduced by V4.
# Root causes diagnosed from V4 pilot (macro F1=0.2947, below V3=0.3084):
#   - TUM collapsed (0.36->0.10): Rule 2 ("orderly crypts->NORM") caused TUM glands
#     to be read as NORM crypts. Fix: add Rule 4 distinguishing TUM from NORM glands
#     and add a CHOOSE TUM directive.
#   - ADI collapsed (0.39->0.11): Step 1 omitted "rounded vacuoles" from the structural
#     feature list, so adipose was eliminated by the chain-of-thought and defaulted to
#     smooth muscle. Fix: add rounded vacuoles to Step 1 and add CHOOSE ADI directive.
#   - DEB low (recall=0.04): same Step 1 gap. Fix: add amorphous necrotic material to
#     Step 1 and add CHOOSE DEB directive.
#   - MUS still over-predicted (precision=0.14): no positive requirement for fascicles.
#     Fix: change MUS from positive description to CHOOSE-only-when-fascicles-present.
# NORM, MUC, STR improvements from V4 are preserved (Rules 1-3 unchanged).
_V4B = (
    "You are an expert gastrointestinal pathologist analyzing an H&E stained "
    "colorectal tissue patch.\n\n"
    "Before choosing a label, reason through these steps. "
    "Do not write the reasoning steps in your response.\n"
    "Step 1: Identify the dominant structural feature: organized glands or crypts, "
    "dense lymphocyte sheets, acellular mucin pools, spindle-cell fibers, "
    "rounded empty vacuoles (adipose), or amorphous necrotic material (debris).\n"
    "Step 2: Eliminate classes whose required features are absent.\n"
    "Step 3: Apply the disambiguation rules below to resolve remaining candidates.\n\n"
    "Tissue classes and their required diagnostic features:\n"
    "- adipose: large empty rounded vacuoles in a honeycomb pattern, thin membranes; "
    "CHOOSE adipose when rounded vacuoles dominate the patch, even if a few compressed "
    "nuclei are visible at cell periphery. Adipose is NOT background (which has no "
    "cellular structure) and NOT mucus (which shows amorphous pools, not round vacuoles).\n"
    "- background: white or near-empty glass, no tissue, no cells\n"
    "- debris: amorphous necrotic material, cellular ghosts, no organized structure; "
    "CHOOSE debris when the patch shows necrotic material without intact glands, "
    "crypts, or organized cell sheets.\n"
    "- lymphocytes: dense sheets of small round dark nuclei, scant cytoplasm\n"
    "- mucus: pale acellular or sparsely cellular pools of extracellular mucin; "
    "blue-gray homogeneous pools with very few intact cells; "
    "CHOOSE mucus when extracellular mucin pools cover most of the patch.\n"
    "- smooth muscle: CHOOSE smooth muscle only when you see organized parallel fascicles "
    "of uniformly elongated spindle cells with blunt cigar-shaped nuclei AND the patch "
    "has no glandular context whatsoever. Smooth muscle requires both positive evidence "
    "(fascicles) and negative evidence (no glands).\n"
    "- normal colon mucosa: regular symmetric crypts, columnar epithelium, goblet cells, "
    "preserved crypt architecture, no nuclear pleomorphism, no invasion; "
    "CHOOSE normal colon mucosa when crypts are present and regular with maintained polarity.\n"
    "- cancer-associated stroma: spindle-cell-rich reactive fibrous tissue adjacent to "
    "or infiltrating malignant glands; loose irregular collagen, reactive fibroblasts; "
    "CHOOSE cancer-associated stroma only when reactive fibrous tissue is in contact "
    "with malignant glands. Without glandular context, prefer smooth muscle.\n"
    "- colorectal adenocarcinoma epithelium: irregular crowded glands, nuclear pleomorphism, "
    "loss of polarity, invasive architecture; "
    "CHOOSE colorectal adenocarcinoma epithelium when glands are present but irregular, "
    "crowded, or show nuclear atypia.\n\n"
    "Disambiguation rules:\n"
    "Rule 1 (smooth muscle vs cancer-associated stroma): Both have spindle cells. "
    "Smooth muscle requires organized parallel fascicles with NO glandular context. "
    "Cancer-associated stroma occurs only adjacent to malignant glands. "
    "Without visible glands, prefer smooth muscle.\n"
    "Rule 2 (normal colon mucosa vs cancer-associated stroma): Normal colon mucosa "
    "has well-organized regular crypts with columnar epithelium. "
    "Cancer-associated stroma has no organized crypts. "
    "If you see orderly crypts, it is normal colon mucosa.\n"
    "Rule 3 (mucus vs normal colon mucosa): Mucus patches are dominated by pale "
    "extracellular pools with sparse cells and no organized crypts. "
    "If the patch shows pale pooled material without intact crypts, choose mucus.\n"
    "Rule 4 (colorectal adenocarcinoma epithelium vs normal colon mucosa): Both show "
    "glandular structures. Tumor glands are irregular, crowded, and show nuclear atypia. "
    "Normal crypts are regular, symmetric, and show maintained cellular polarity. "
    "If glands are present and irregular or crowded, it is colorectal adenocarcinoma "
    "epithelium, not normal colon mucosa.\n\n"
    "{clinical_context}\n\n"
    "Respond in exactly this format and nothing else:\n"
    "LABEL: <one tissue type from the list above>\n"
    "CONFIDENCE: <integer 0-100>\n"
    "REASONING: <one sentence naming the dominant histological features you observed>"
)

# V5: clean redesign after V4/V4B failure analysis.
# Root cause of V4/V4B failures: "CHOOSE X when Y" directives cause MedGemma at
# temperature=0.0 to latch onto the first high-salience positive signal globally.
# V4 over-predicted NORM (CHOOSE-NORM appeared first with strong criteria).
# V4B over-predicted ADI (CHOOSE-ADI appeared first in class list).
# V5 principle: no CHOOSE directives anywhere. Balanced class descriptions at the
# same assertiveness level. Four disambiguation rules stated as observations, not
# commands. Brief single-sentence reasoning instruction without multi-step structure.
# Target: recover V3 strengths (ADI=0.39, LYM=0.67, BACK=0.72, TUM=0.36) while
# fixing V3 zero-F1 classes (NORM, MUC, STR) through better descriptions + rules.
_V5 = (
    "You are an expert gastrointestinal pathologist analyzing an H&E stained "
    "colorectal tissue patch.\n\n"
    "Identify the dominant structural feature in the patch before selecting a label. "
    "Do not include your reasoning in the response.\n\n"
    "Tissue classes:\n"
    "- adipose: large empty rounded vacuoles arranged in a honeycomb pattern, "
    "thin membranes, no glandular or fibrous architecture\n"
    "- background: empty glass slide, no tissue, no cells\n"
    "- debris: amorphous necrotic material, cellular ghosts, no organized structure\n"
    "- lymphocytes: dense sheets of small round dark-nuclei cells with minimal cytoplasm\n"
    "- mucus: pale pools of extracellular mucin as the dominant feature, "
    "very few intact cells, no organized crypts visible\n"
    "- smooth muscle: organized parallel fascicles of elongated spindle cells "
    "with blunt cigar-shaped nuclei, abundant eosinophilic cytoplasm\n"
    "- normal colon mucosa: regular symmetric crypts lined by columnar epithelium "
    "with goblet cells, maintained polarity, no invasion\n"
    "- cancer-associated stroma: reactive spindle-cell fibrous tissue immediately "
    "adjacent to or surrounding malignant glandular structures, desmoplastic matrix\n"
    "- colorectal adenocarcinoma epithelium: irregular crowded malignant glands "
    "with nuclear pleomorphism and loss of polarity\n\n"
    "Apply these distinctions when two classes remain as candidates:\n"
    "1. Smooth muscle vs cancer-associated stroma: smooth muscle has organized parallel "
    "fascicles with no glandular structures anywhere in the patch. Cancer-associated "
    "stroma has fibrous tissue in contact with malignant glands. No glands visible "
    "favors smooth muscle.\n"
    "2. Normal colon mucosa vs cancer-associated stroma: normal colon mucosa has "
    "regular organized crypts with columnar epithelium. Cancer-associated stroma has "
    "no organized crypts. Regular crypts favor normal colon mucosa.\n"
    "3. Mucus vs normal colon mucosa: mucus shows pale extracellular pools with no "
    "organized crypts. Pale extracellular material with sparse cells and no crypts "
    "favors mucus over normal colon mucosa.\n"
    "4. Colorectal adenocarcinoma epithelium vs normal colon mucosa: both show "
    "glandular structures. Tumor glands are irregular and crowded with nuclear atypia. "
    "Normal crypts are regular and symmetric. Irregular or crowded glands favor tumor.\n\n"
    "{clinical_context}\n\n"
    "Respond in exactly this format and nothing else:\n"
    "LABEL: <one tissue type from the list above>\n"
    "CONFIDENCE: <integer 0-100>\n"
    "REASONING: <one sentence naming the key distinguishing feature you observed>"
)

_VERSIONS = {"V1": _V1, "V2": _V2, "V3": _V3, "V4": _V4, "V4B": _V4B, "V5": _V5}


def get_prompt(task: str, version: str = "V2", clinical_context: str = "") -> str:
    """
    Build a prompt for a task and version.

    For the tissue task, `task` is ignored beyond validation; the same prompt
    body covers the nine-class problem. The parameter is kept so the signature
    matches client._default_prompt and so new tasks can be added later.

    Args:
        task: Task key. Currently "tissue_classification" (or any value; the
            tissue prompt is returned). Reserved for future task families.
        version: One of "V1", "V2", "V3", "V4".
        clinical_context: Optional metadata string injected into V2/V3/V4.

    Returns:
        The formatted prompt string.
    """
    if version not in _VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}. Use one of {list(_VERSIONS)}.")

    template = _VERSIONS[version]
    ctx = f"Clinical context: {clinical_context}" if clinical_context else ""
    # V1 has no context slot; format() on it is a no-op.
    if "{clinical_context}" in template:
        return template.format(clinical_context=ctx)
    return template
