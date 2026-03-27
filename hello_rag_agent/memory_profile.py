from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ProfileFact:
    fact_key: str
    fact_value: str
    confidence: float


PROFILE_LABELS: dict[str, str] = {
    "floor_type": "地面",
    "pets": "宠物",
    "home_size": "户型",
    "noise_preference": "偏好",
    "app_stability_priority": "偏好",
    "anti_tangle_priority": "偏好",
    "carpet_presence": "地毯",
    "maintenance_status": "维护状态",
    "recent_issue": "问题",
}


def format_profile_fact_line(fact_key: str, fact_value: str) -> str:
    label = PROFILE_LABELS.get(fact_key, fact_key)
    return f"{label}: {fact_value}"


def extract_profile_facts(text: str) -> list[ProfileFact]:
    normalized = text.strip()
    if not normalized:
        return []

    facts: list[ProfileFact] = []
    facts.extend(_extract_floor_type(normalized))
    facts.extend(_extract_pets(normalized))
    facts.extend(_extract_home_size(normalized))
    facts.extend(_extract_preferences(normalized))
    facts.extend(_extract_carpet_presence(normalized))
    facts.extend(_extract_maintenance_status(normalized))

    deduped: dict[tuple[str, str], ProfileFact] = {}
    for fact in facts:
        key = (fact.fact_key, fact.fact_value)
        previous = deduped.get(key)
        if previous is None or fact.confidence > previous.confidence:
            deduped[key] = fact
    return list(deduped.values())


def _extract_floor_type(text: str) -> list[ProfileFact]:
    mappings = (
        ("木地板", "木地板"),
        ("瓷砖", "瓷砖"),
        ("地砖", "地砖"),
        ("大理石", "大理石"),
    )
    return [
        ProfileFact("floor_type", value, 0.95)
        for keyword, value in mappings
        if keyword in text
    ]


def _extract_pets(text: str) -> list[ProfileFact]:
    count_match = re.search(r"(?:养了|家里有|有)([一二两三四五六七八九十0-9]+)只", text)
    count_text = count_match.group(1) if count_match else ""

    pet_value = ""
    confidence = 0.0

    if "长毛猫" in text:
        pet_value = "长毛猫"
        confidence = 0.97
    elif "短毛猫" in text:
        pet_value = "短毛猫"
        confidence = 0.96
    elif "猫" in text:
        pet_value = "猫"
        confidence = 0.9
    elif "狗" in text:
        pet_value = "狗"
        confidence = 0.9

    if not pet_value:
        return []

    if count_text:
        pet_value = f"{count_text}只{pet_value}"
        confidence = min(max(confidence, 0.92), 0.98)

    return [ProfileFact("pets", pet_value, confidence)]


def _extract_home_size(text: str) -> list[ProfileFact]:
    facts: list[ProfileFact] = []
    area_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:平米|平)", text)
    if area_match:
        facts.append(ProfileFact("home_size", f"{area_match.group(1)}平", 0.95))
    if "大户型" in text:
        facts.append(ProfileFact("home_size", "大户型", 0.9))
    if "小户型" in text:
        facts.append(ProfileFact("home_size", "小户型", 0.9))
    return facts


def _extract_preferences(text: str) -> list[ProfileFact]:
    facts: list[ProfileFact] = []
    if "静音" in text:
        facts.append(ProfileFact("noise_preference", "更在意静音", 0.95))
    if "APP稳定" in text or "app稳定" in text.lower():
        facts.append(ProfileFact("app_stability_priority", "更在意APP稳定性", 0.95))
    if "防缠绕" in text:
        facts.append(ProfileFact("anti_tangle_priority", "更在意防缠绕", 0.92))
    return facts


def _extract_carpet_presence(text: str) -> list[ProfileFact]:
    facts: list[ProfileFact] = []
    if "没有地毯" in text or "无地毯" in text:
        facts.append(ProfileFact("carpet_presence", "没有地毯", 0.9))
    elif "地毯" in text:
        facts.append(ProfileFact("carpet_presence", "有地毯", 0.85))
    return facts


def _extract_maintenance_status(text: str) -> list[ProfileFact]:
    facts: list[ProfileFact] = []
    if "洗过过滤网" in text or "清洗过滤网" in text:
        facts.append(ProfileFact("maintenance_status", "已清洗过滤网", 0.9))
    if "滚刷还没清理" in text or "滚刷还未清理" in text:
        facts.append(ProfileFact("maintenance_status", "滚刷还没清理", 0.9))
    if "边刷" in text and ("没更换" in text or "还没换" in text or "未更换" in text):
        facts.append(ProfileFact("maintenance_status", "边刷还没更换", 0.88))
    if "APP" in text and ("掉线" in text or "断连" in text or "不稳定" in text):
        facts.append(ProfileFact("recent_issue", "APP偶尔掉线", 0.88))
    if "回充" in text and ("失败" in text or "异常" in text):
        facts.append(ProfileFact("recent_issue", "回充偶尔失败", 0.88))
    return facts
