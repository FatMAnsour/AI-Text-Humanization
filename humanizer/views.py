from django.shortcuts import render
from openai import OpenAI
from django.conf import settings
import random
import re

try:
    from humanizer.style_stats import get_style_target_prompt
except ImportError:
    def get_style_target_prompt():
        return ""


def _get_groq_client():
    return OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    ), "llama-3.3-70b-versatile"


def _get_mistral_client():
    if not getattr(settings, "MISTRAL_API_KEY", None):
        return None, None
    client = OpenAI(
        api_key=settings.MISTRAL_API_KEY,
        base_url="https://api.mistral.ai/v1",
    )
    model = getattr(settings, "MISTRAL_HUMANIZE_MODEL", "mistral-large-latest")
    return client, model


def _get_cohere_client():
    if not getattr(settings, "COHERE_API_KEY", None):
        return None, None
    client = OpenAI(
        api_key=settings.COHERE_API_KEY,
        base_url="https://api.cohere.ai/compatibility/v1",
    )
    model = getattr(settings, "COHERE_HUMANIZE_MODEL", "command-a-03-2025")
    return client, model


def _call_llm_with(
    client,
    model,
    prompt: str,
    temperature: float = 0.92,
    top_p: float = 0.93,
    system_message: str = None,
    presence_penalty: float = 0.3,
    frequency_penalty: float = 0.4,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=3200,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    out = response.choices[0].message.content.strip()
    if out.startswith('"') and out.endswith('"') and len(out) > 1:
        out = out[1:-1].strip()
    return out


STYLE_HINTS = {
    "casual": "Friendly and conversational with plain, everyday wording.",
    "professional": "Professional and direct, but still warm and readable.",
    "blog": "Readable and opinionated, with natural flow and personality.",
    "story": "Narrative style with concrete details and scene-like flow.",
}

AUDIENCE_HINTS = {
    "general": "general audience",
    "student": "students",
    "client": "clients or stakeholders",
    "expert": "subject-matter experts",
}

# Words/phrases that trigger AI detectors. Never use these (research: ChadGPT + PureWrite + Reddit + detector lists).
BANNED_PHRASES = (
    "Furthermore, Moreover, In conclusion, It's important to note, It is worth noting, delve, leverage, crucial, "
    "comprehensive, utilize, paradigm, nuanced, facilitate, In order to, Moving forward, In terms of, "
    "With that said, Having said that, cutting-edge, state-of-the-art, best practices, synergy, "
    "in today's world, unlock your potential, as a result, For instance, In other words, "
    "It goes without saying, At the end of the day, When it comes to, plays a key role, "
    "embark, shed light, tapestry, illuminate, unveil, pivotal, intricate, elucidate, hence, harness, "
    "groundbreaking, remarkable, moreover, skyrocket, realm, disruptive, navigate, landscape, testament, "
    "not only... but also, in summary, in closing, in the dynamic world, ever-evolving, game-changer, "
    "It is essential, In today's society, In the modern world, As previously stated, It should be noted, "
    "diving deeper, multifaceted, holistic, streamline, robust, seamless, optimize, enhance, implement, "
    "first and foremost, at the end of the day, when it comes down to it, in essence, to sum up."
)

# In-context human-style examples: high burstiness, casual tone, fragments, contractions (target detector evasion).
HUMAN_STYLE_EXAMPLES = [
    "So I tried that and it actually worked. Took a while though. And the main thing is you have to do it every day or it doesn't stick.",
    "Honestly it depends. Sometimes you get lucky and sometimes you don't. I mean that's just how it goes.",
    "Yeah it's not perfect. But we're getting there. And that's what matters in the end.",
    "It's one of those things. You think you're done and then nope. Anyway that's the idea.",
    "The problem is nobody really knows. Like you try one thing and it works for someone else but not for you. So you just have to test it.",
    "I looked it up and yeah the results were mixed. Some people said do this, others said do that. In the end I went with what felt right.",
]


def _get_anti_detect_system(style_hint: str = "", audience_hint: str = ""):
    """
    Detector-aware humanizer: GPTZero/ZeroGPT classify by perplexity + burstiness.
    Goal: output must be classified as human (high perplexity, high burstiness, informal markers).
    """
    target = get_style_target_prompt()
    example = random.choice(HUMAN_STYLE_EXAMPLES)
    base = (
        "CRITICAL: Your output will be checked by AI detectors (e.g. GPTZero, ZeroGPT). "
        "You MUST produce text they classify as HUMAN. Human writing has: unpredictable word choice (avoid the most obvious next word), "
        "uneven sentence lengths (mix very short fragments with longer sentences), and small imperfections. "
        "Rewrite the user's text so it reads like a real person typed it in one go. Same meaning, similar length. Output ONLY the rewritten text. "
        "RULES: "
        "(1) Vary sentence length a lot. Include at least 2–3 very short sentences (under 6 words) and at least 1 longer one (18+ words). "
        "(2) Use contractions (I'm, don't, it's, that's, you're). Active voice. "
        "(3) NO em dashes (—) or semicolons. Only commas and periods. "
        "(4) NO 'not only X but also Y', no tricolons, no parallel lists of three. "
        "(5) Use 'you' or 'your' where natural. Be specific, not generic. "
        "(6) BANNED (never use): " + BANNED_PHRASES + " "
        "(7) No bullet points, no lists, no markdown. No 'In conclusion' or 'In summary'. Plain prose only. "
        "(8) Start at least one sentence with And or But. One fragment (e.g. 'Took a while.') or run-on is fine. One casual opener (Well, So, Honestly, I mean) is fine. "
        "Write in this kind of rhythm: \"" + example + "\""
    )
    if style_hint:
        base = f"Style: {style_hint}. " + base
    if audience_hint:
        base = f"Audience: {audience_hint}. " + base
    if target:
        base += " " + target
    return base


SINGLE_PASS_PROMPT = """Rewrite this so it sounds like a real person typing (not a polished essay). Same meaning, similar length.
Use clear, simple language. Mix very short sentences with a few longer ones. Contractions. Active voice. No em dashes or semicolons. No lists. Style: {style_hint}. Output only the rewritten text.

{text}"""

# Pass 1: break original structure and wording (different words + structure).
PASS1_USER = """Rewrite this in completely different words and sentence structure. Same meaning, similar length.
Include at least 2 very short sentences (under 6 words) and 1 longer (18+ words). No em dashes. Output only the text.

{text}"""

# Pass 2: detector-aware humanizer; system has full banned list and rhythm example.
PASS2_USER = """Rewrite so it reads like a real person wrote it. Same meaning and length. Short and long sentences, contractions, active voice. No em dashes or semicolons. Style: {style_hint}. Output only the text.

{text}"""

# Key-point rewrite: break attachment to original phrasing (max structure break for detectors).
KEYPOINTS_USER = """List ONLY the main ideas below as 5–8 short phrases. Each phrase 2–5 words. One per line. No full sentences, no periods at end of phrases.

{text}"""

KEYPOINTS_TO_PROSE_USER = """From ONLY these phrases, write one short paragraph. Write like someone typing quickly: short sentences, some fragments, contractions (it's, don't, that's). Vary length a lot (some under 5 words, one or two longer). No em dashes, no semicolons. No lists or bullets. Output only the paragraph. It must read like casual human writing, not polished.

Points:
{text}"""

# Stack: rephrase with strong humanizer rules; final pass highest temperature.
STACK_PASS1 = (
    "Rewrite in completely different words and structure. Same meaning, similar length. "
    "Include 2+ very short sentences and 1 longer. No em dashes. Output only the text.\n\n{text}"
)
STACK_PASS2 = (
    "Make this sound like a real person typing. Same meaning. Contractions, active voice, mix short and long sentences. "
    "No em dashes or semicolons. Output only the text.\n\n{text}"
)
STACK_PASS3 = (
    "Final pass: must read as human-written. Same meaning. Very short and longer sentences, contractions, no polish. "
    "No em dashes, semicolons, or formal words. Output only the text.\n\n{text}"
)


def _postprocess_cleanup(text: str) -> str:
    if not text:
        return text
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _postprocess_noise(text: str) -> str:
    """Light non-LLM variation: occasional filler, fragment, or run-on. Keeps output less uniform for evasion."""
    if not text:
        return text
    fillers = ["Well,", "So,", "I mean,"]

    def split_sentences(chunk: str):
        parts = re.findall(r".*?(?:[.!?]+|$)", chunk, flags=re.DOTALL)
        return [p for p in parts if p.strip()]

    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            processed_lines.append(line)
            continue
        sentences = split_sentences(stripped)
        new_sentences = []
        for s in sentences:
            s_strip = s.strip()
            if not s_strip:
                continue
            if random.random() < 0.07 and len(s_strip) > 10:
                filler = random.choice(fillers)
                if len(s_strip) > 1:
                    s_strip = f"{filler} {s_strip[0].lower()}{s_strip[1:]}"
                else:
                    s_strip = f"{filler} {s_strip}"
            if random.random() < 0.08:
                s_strip = re.sub(r"[.!?]+$", "", s_strip)
            if random.random() < 0.05 and "," in s_strip and len(s_strip) > 20:
                s_strip = s_strip.replace(",", "", 1)
            new_sentences.append(s_strip)
        i = 0
        merged = []
        while i < len(new_sentences):
            if i < len(new_sentences) - 1 and random.random() < 0.12:
                merged.append(new_sentences[i].rstrip(" .!?") + " and " + new_sentences[i + 1].lstrip())
                i += 2
            else:
                merged.append(new_sentences[i])
                i += 1
        processed_lines.append(" ".join(merged))
    return "\n".join(processed_lines)


FORMAL_REPLACEMENTS = {
    r"\bin conclusion\b": "to wrap up",
    r"\bit is important to note that\b": "",
    r"\bmoreover\b": "also",
    r"\bfurthermore\b": "also",
    r"\bin order to\b": "to",
    r"\butilize\b": "use",
    r"\bleverage\b": "use",
    r"\bwith that said\b": "that said",
    r"\bmoving forward\b": "next",
    r"\bbest practices\b": "practical steps",
    r"\bfacilitate\b": "help",
    r"\bhowever\b": "but",
}


def _strip_ai_punctuation(text: str) -> str:
    """Remove em dashes and replace semicolons with periods (AI tells)."""
    if not text:
        return text
    text = re.sub(r"—", ", ", text)
    text = re.sub(r"\s*;\s*", ". ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _soften_formal_phrases(text: str) -> str:
    if not text:
        return text
    updated = text
    for pattern, replacement in FORMAL_REPLACEMENTS.items():
        updated = re.sub(pattern, replacement, updated, flags=re.IGNORECASE)
    # Remove doubled spaces introduced by replacements.
    updated = re.sub(r"[ \t]{2,}", " ", updated)
    return updated.strip()


def _strip_editor_notes(text: str) -> str:
    if not text:
        return text
    # Remove common trailing note blocks ("I made the following changes...", bullet lists).
    marker_pattern = re.compile(
        r"(?is)\n+\s*(i made the following changes|here are the changes|changes made:|what i changed)\b.*$"
    )
    stripped = marker_pattern.sub("", text).strip()
    stripped = re.sub(r"(?is)^```(?:text)?\s*", "", stripped).strip()
    stripped = re.sub(r"(?is)\s*```$", "", stripped).strip()
    # Remove common leading preambles before the real output.
    leading_patterns = [
        r"(?im)^\s*here(?:'s| is)\s+(?:a\s+)?revised version(?:\s+with.*)?\s*:\s*",
        r"(?im)^\s*revised version(?:\s+with.*)?\s*:\s*",
        r"(?im)^\s*here(?:'s| is)\s+the rewrite\s*:\s*",
        r"(?im)^\s*rewritten text\s*:\s*",
    ]
    for pattern in leading_patterns:
        stripped = re.sub(pattern, "", stripped, count=1).strip()
    return stripped or text.strip()


def humanize_view(request):
    result = None
    error = None
    if request.method == "POST":
        text = request.POST.get("text", "").strip()
        style = request.POST.get("style", "casual").strip().lower()
        audience = request.POST.get("audience", "general").strip().lower()
        purpose = request.POST.get("purpose", "inform and engage").strip()

        style_hint = STYLE_HINTS.get(style, STYLE_HINTS["casual"])
        audience_hint = AUDIENCE_HINTS.get(audience, AUDIENCE_HINTS["general"])
        purpose = purpose or "inform and engage"

        if text:
            try:
                provider = getattr(settings, "HUMANIZE_PROVIDER", "groq")
                groq_client, groq_model = _get_groq_client()
                anti_system = _get_anti_detect_system(style_hint, audience_hint)

                if provider == "stack4":
                    current = _call_llm_with(
                        groq_client,
                        groq_model,
                        STACK_PASS1.format(text=text),
                        0.92,
                        0.93,
                        system_message=None,
                        presence_penalty=0.3,
                        frequency_penalty=0.4,
                    )
                    mistral_client, mistral_model = _get_mistral_client()
                    if mistral_client and mistral_model:
                        try:
                            current = _call_llm_with(
                                mistral_client,
                                mistral_model,
                                STACK_PASS2.format(text=current),
                                0.94,
                                0.94,
                                system_message=anti_system,
                                presence_penalty=0.35,
                                frequency_penalty=0.45,
                            )
                        except Exception:
                            pass
                    cohere_client, cohere_model = _get_cohere_client()
                    if cohere_client and cohere_model:
                        try:
                            current = _call_llm_with(
                                cohere_client,
                                cohere_model,
                                STACK_PASS3.format(text=current),
                                0.98,
                                0.96,
                                system_message=anti_system,
                                presence_penalty=0.45,
                                frequency_penalty=0.55,
                            )
                        except Exception:
                            pass
                    humanized = _postprocess_noise(current)
                else:
                    passes = max(1, min(2, getattr(settings, "HUMANIZE_PASSES", 2)))
                    use_keypoints = getattr(settings, "HUMANIZE_KEYPOINT_REWRITE", True) and passes == 2
                    if passes == 1:
                        humanized = _call_llm_with(
                            groq_client,
                            groq_model,
                            SINGLE_PASS_PROMPT.format(style_hint=style_hint, text=text),
                            0.94,
                            0.93,
                            system_message=None,
                            presence_penalty=0.28,
                            frequency_penalty=0.38,
                        )
                    elif use_keypoints:
                        points = _call_llm_with(
                            groq_client,
                            groq_model,
                            KEYPOINTS_USER.format(text=text),
                            0.92,
                            0.92,
                            system_message=None,
                            presence_penalty=0.25,
                            frequency_penalty=0.35,
                        )
                        humanized = _call_llm_with(
                            groq_client,
                            groq_model,
                            KEYPOINTS_TO_PROSE_USER.format(text=points.strip()),
                            0.98,
                            0.96,
                            system_message=anti_system,
                            presence_penalty=0.42,
                            frequency_penalty=0.52,
                        )
                    else:
                        pass1 = _call_llm_with(
                            groq_client,
                            groq_model,
                            PASS1_USER.format(text=text),
                            0.92,
                            0.93,
                            system_message=None,
                            presence_penalty=0.28,
                            frequency_penalty=0.38,
                        )
                        humanized = _call_llm_with(
                            groq_client,
                            groq_model,
                            PASS2_USER.format(text=pass1, style_hint=style_hint),
                            0.96,
                            0.95,
                            system_message=anti_system,
                            presence_penalty=0.38,
                            frequency_penalty=0.48,
                        )
                    humanized = _postprocess_noise(humanized)

                humanized = _postprocess_cleanup(_strip_editor_notes(humanized))
                humanized = _soften_formal_phrases(humanized)
                humanized = _strip_ai_punctuation(humanized)

                result = {
                    "original": text,
                    "humanized": humanized,
                    "style": style,
                    "audience": audience,
                    "purpose": purpose,
                    "original_words": len(text.split()),
                    "humanized_words": len(humanized.split()),
                }
            except Exception as e:
                error = f"Error: {str(e)} (maybe rate limit - wait 1 minute and try again)"

    return render(
        request,
        "humanizer/humanize.html",
        {
            "result": result,
            "error": error,
            "style_options": STYLE_HINTS,
            "audience_options": AUDIENCE_HINTS,
        },
    )


def _test_one_api(name: str, client, model: str) -> tuple[bool, str]:
    """Try a minimal completion. Returns (success, message)."""
    if client is None or model is None:
        return False, "No API key set"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with only the word OK."}],
            max_tokens=5,
            temperature=0,
        )
        out = (response.choices[0].message.content or "").strip()
        return True, "OK" if out else "Empty response"
    except Exception as e:
        return False, str(e)


def api_status_view(request):
    """Check that all configured APIs are reachable and working."""
    results = []

    for name, get_client, key_attr in [
        ("Groq", _get_groq_client, "GROQ_API_KEY"),
        ("Mistral", _get_mistral_client, "MISTRAL_API_KEY"),
        ("Cohere", _get_cohere_client, "COHERE_API_KEY"),
    ]:
        key_set = bool(getattr(settings, key_attr, None))
        client, model = get_client()
        if client and model:
            ok, msg = _test_one_api(name, client, model)
        else:
            ok, msg = False, "No API key set" if not key_set else "Missing key"
        results.append({"name": name, "key_set": key_set, "ok": ok, "message": msg})

    return render(request, "humanizer/api_status.html", {"results": results})
