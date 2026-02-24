from core.prefs_infer import infer_preferences_from_text


def _as_map(items):
    return {k: v for k, v in items}


def test_infer_language_ru_uk_en():
    ru = _as_map(infer_preferences_from_text("Пиши на русском языке"))
    uk = _as_map(infer_preferences_from_text("Відповідай українською мовою"))
    en = _as_map(infer_preferences_from_text("Please answer in english language"))

    assert ru.get("style.language") == "ru"
    assert uk.get("style.language") == "uk"
    assert en.get("style.language") == "en"


def test_infer_code_language_from_ru_uk_en_prompts():
    ru = _as_map(infer_preferences_from_text("комментарии в коде на английском"))
    uk = _as_map(infer_preferences_from_text("коментарі в коді англійською"))
    en = _as_map(infer_preferences_from_text("comments in code should be english"))

    assert ru.get("style.code_comments_language") == "en"
    assert uk.get("style.code_comments_language") == "en"
    assert en.get("style.code_comments_language") == "en"


def test_infer_block_tlds_and_currency_multilingual():
    ru = _as_map(infer_preferences_from_text("не предлагать ресурсы из доменов *.ru"))
    uk = _as_map(infer_preferences_from_text("не пропонувати ресурси з доменів *.ru"))
    en = _as_map(infer_preferences_from_text("do not suggest resources from domains *.ru"))
    cur = _as_map(infer_preferences_from_text("ціни в гривні"))

    assert "web.block_tlds" in ru
    assert "рф" in ru["web.block_tlds"]
    assert "web.block_tlds" in uk
    assert "web.block_tlds" in en
    assert cur.get("finance.currency") == "UAH"
