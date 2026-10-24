from core.ai.text_client import text_client
from core.prompts.intent_prompts import INTENT_EXTRACTION_PROMPT_TEMPLATE
from shared.logger import get_logger
from shared.exceptions import AIAnalysisError
from shared.models.enums import OutfitOccasion, OutfitMood
from shared.models.intent import ExtractedIntent

logger = get_logger(__name__)

MIN_CONFIDENCE = 0.4

# Appended to the prompt so the model always rates itself. Without it some
# answers came back with no usable "confidence", which became 0.0.
CONFIDENCE_INSTRUCTION = (
    '\n\nAlways include a "confidence" field: a number between 0.0 and 1.0 '
    "for how sure you are about the occasion. Return only the JSON object."
)

# Some answers use words instead of numbers.
WORD_CONFIDENCE = {"very high": 0.95, "high": 0.9, "medium": 0.6, "moderate": 0.6,
                   "low": 0.3, "very low": 0.1}


def extract_intent(user_text: str) -> ExtractedIntent:
    """
    Converts free text ("dinner with my girlfriend tonight") into
    structured occasion/mood the stylist pipeline already understands.
    GPT never sees wardrobe data - this is a pure text-to-intent step,
    isolated from the generator entirely.
    """
    if not user_text or not user_text.strip():
        raise AIAnalysisError("Please describe what you need an outfit for.")

    try:
        logger.info("Extracting intent from text: '%s'", user_text[:100])

        prompt = INTENT_EXTRACTION_PROMPT_TEMPLATE.format(user_text=user_text.strip())
        raw = text_client.analyze_text(prompt + CONFIDENCE_INSTRUCTION)

        if "occasion" not in raw:
            logger.error("GPT-4o missing occasion field")
            raise AIAnalysisError("Couldn't understand that request. Please try again.")

        confidence = _parse_confidence(raw)
        if confidence < MIN_CONFIDENCE:
            logger.warning("Low confidence intent extraction: %.2f", confidence)

        occasion = _parse_occasion(raw.get("occasion"))
        mood = _parse_mood(raw.get("mood"))

        logger.info(
            "Intent extracted - occasion=%s mood=%s confidence=%.2f",
            occasion.value, mood.value if mood else None, confidence
        )

        return ExtractedIntent(
            occasion=occasion,
            mood=mood,
            confidence=confidence
        )

    except AIAnalysisError:
        raise
    except Exception:
        logger.exception("Intent extraction failed")
        raise AIAnalysisError("Couldn't understand that request. Please try again.")


def _parse_confidence(raw: dict) -> float:
    value = next(
        (raw[k] for k in ("confidence", "confidence_score", "certainty") if k in raw),
        None,
    )
    if value is None:
        logger.warning("GPT returned no confidence field - keys=%s", list(raw.keys()))
        return 0.0
    if isinstance(value, str) and value.strip().lower() in WORD_CONFIDENCE:
        return WORD_CONFIDENCE[value.strip().lower()]
    try:
        number = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        logger.warning("Invalid confidence value from GPT: %r", value)
        return 0.0
    if number > 1.0:          # "85" or "85%" -> 0.85
        number = number / 100.0
    return max(0.0, min(number, 1.0))


def _parse_occasion(value) -> OutfitOccasion:
    try:
        return OutfitOccasion(str(value).lower())
    except (ValueError, KeyError):
        logger.warning("Unknown occasion '%s' - defaulting to everyday", value)
        return OutfitOccasion.EVERYDAY


def _parse_mood(value) -> OutfitMood | None:
    if not value or str(value).lower() == "null":
        return None
    try:
        return OutfitMood(str(value).lower())
    except (ValueError, KeyError):
        logger.warning("Unknown mood '%s' - ignoring", value)
        return None