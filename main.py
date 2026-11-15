#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, ten free-tier providers.

Serves the OpenAI Chat Completions API (POST /v1/chat/completions) and the
Anthropic Messages API (POST /v1/messages), in JSON and in server-sent
events, routing every request to the best free model that meets it across
ten simulated providers speaking the OpenAI, Anthropic and Gemini wires.

How it works, in one paragraph: the trial hands this server a brief
(MAKEFUN_PROMPT_FILE) with the provider directory, the sealed quality table
and the workload manifest. At startup it probes every provider's own catalog
route, builds one catalog document from what the listings said, registers
its port and that catalog with the env (POST $MAKEFUN_ENV_URL/register),
then answers the client's twenty requests one at a time. Each request is
routed to the highest-rated free model meeting its requirements, spread
across providers under a soft per-provider cap so no single free tier is
drained, and on a 402/429/5xx it fails over to the next-best candidate.
Answers are translated back into the wire the client used, with the
provider's text, usage and stop reason carried over exactly. When the env
says the workload is done (GET $MAKEFUN_ENV_URL/work), the server exits 0.
`python3 main.py --list` prints the catalog and exits 0.

Pure standard library only: http.server, urllib, json, threading. Every
choice is a function of the brief, the requests and the providers' answers
so far: no wall clock, no unseeded randomness, so a replay reproduces the
same run byte for byte.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 25
MAX_BODY_BYTES = 32 * 1024 * 1024
# Max primary calls handed to one provider before the router spreads to the
# next-best model elsewhere. Budgets are drawn per trial and never published;
# four keeps every provider comfortably under its smallest plausible budget
# while the top models still serve most requests.
SOFT_CAP = 4

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


def text_of(value):
    return value if isinstance(value, str) else ""


def number_or_none(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value == value and abs(value) != float("inf"):
        return value
    return None


def bool_or_none(value):
    return value if isinstance(value, bool) else None


# ---------------------------------------------------------------------------
# The brief: three JSON documents found by their `schema` member.
# ---------------------------------------------------------------------------

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
                if is_record(value) and isinstance(value.get("schema"), str):
                    out.append(value)
            except Exception:
                pass
        start = text.find("{", start + 1)
    return out


def _provider_of(raw):
    if not is_record(raw):
        return None
    pid = text_of(raw.get("id"))
    wire = raw.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = raw.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if text_of(raw.get(key)) != "":
            base = text_of(raw.get(key))
            break
    return {
        "id": pid,
        "name": text_of(raw.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": text_of(raw.get("key_env")),
        "key": text_of(raw.get("key")),
    }


def _rating_of(raw):
    if not is_record(raw):
        return None
    model = text_of(raw.get("model"))
    provider = text_of(raw.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": number_or_none(raw.get("rating_milli")) or 0,
        "context_length": number_or_none(raw.get("context_length")),
        "tools": raw.get("tools") is True,
        "vision": raw.get("vision") is True,
        "json": raw.get("json") is True,
        "free": raw.get("free") is True,
    }


def _workload_request_of(raw):
    if not is_record(raw):
        return None
    rid = text_of(raw.get("id"))
    if rid == "":
        return None
    req = raw.get("requires") if is_record(raw.get("requires")) else {}
    out = {"id": rid, "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
           "stream": raw.get("stream") is True, "weight": 2 if raw.get("weight") == 2 else 1,
           "requires": {}}
    if req.get("tools") is True:
        out["requires"]["tools"] = True
    if req.get("vision") is True:
        out["requires"]["vision"] = True
    if req.get("json") is True:
        out["requires"]["json"] = True
    mc = number_or_none(req.get("min_context"))
    if mc is not None:
        out["requires"]["min_context"] = mc
    return out


def read_brief(path):
    brief = {"providers": [], "ratings": [], "best": {}, "workload": [], "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
                      and isinstance(d.get("providers"), list)), None)
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        provider = _provider_of(raw)
        if provider is not None:
            brief["providers"].append(provider)
    ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA
                        and isinstance(d.get("ratings"), list)), None)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            row = _rating_of(raw)
            if row is not None:
                brief["ratings"].append(row)
        if isinstance(ratings_doc.get("best"), list):
            for raw in ratings_doc["best"]:
                if is_record(raw) and text_of(raw.get("id")) != "":
                    brief["best"][raw["id"]] = number_or_none(raw.get("rating_milli")) or 0
    workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
                         and isinstance(d.get("manifest"), list)), None)
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            entry = _workload_request_of(raw)
            if entry is not None:
                brief["workload"].append(entry)
    return brief


# ---------------------------------------------------------------------------
# Providers: auth headers, catalog probes, the catalog document.
# ---------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if re.match(r"^https?://", b, re.IGNORECASE):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(raw):
    pricing = raw.get("pricing") if is_record(raw.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for value in pricing.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(value)
        elif isinstance(value, str) and value.strip() != "":
            try:
                values.append(float(value))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def _model_in(wire, raw):
    if not is_record(raw):
        return None
    if wire == "gemini":
        mid = text_of(raw.get("name")).removeprefix("models/")
    else:
        mid = text_of(raw.get("id"))
    if mid == "":
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        context = number_or_none(raw.get(key))
        if context is not None:
            break
    free = bool_or_none(raw.get("free"))
    if free is None:
        free = _zero_priced(raw)
    return {
        "id": mid,
        "display_name": text_of(raw.get("display_name")) or text_of(raw.get("displayName")) or mid,
        "free": free,
        "context_length": context,
        "tools": bool_or_none(raw.get("tools")),
        "vision": bool_or_none(raw.get("vision")),
        "json": bool_or_none(raw.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    listing = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return []
    out = []
    for raw in listing:
        model = _model_in(wire, raw)
        if model is not None:
            out.append(model)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return {"provider": provider, "ok": False, "models": [],
                        "error": "GET %s answered HTTP %d" % (url, response.status)}
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"provider": provider, "ok": False, "models": [],
                "error": "GET %s failed: %s" % (url, exc)}
    return {"provider": provider, "ok": True, "models": _models_in(provider["wire"], body),
            "error": None}


class Catalog:
    """The brief, the probes, the ratings and the routing state, in one place."""

    def __init__(self, brief, env_url):
        self.brief = brief
        self.env_url = env_url
        self.probes = [probe_provider(env_url, p) for p in brief["providers"]]
        for probe in self.probes:
            if not probe["ok"] and probe["error"]:
                log(probe["error"])
        self.ratings = {(r["provider"], r["model"]): r for r in brief["ratings"]}
        self.provider_order = {p["id"]: i for i, p in enumerate(brief["providers"])}
        self.document = self._build_document()
        self.primary_use = {p["id"]: 0 for p in brief["providers"]}

    def rating_row(self, provider_id, model_id):
        return self.ratings.get((provider_id, model_id))

    def _build_document(self):
        providers = []
        for probe in self.probes:
            if not probe["ok"]:
                continue
            seen = set()
            models = []
            for model in sorted(probe["models"], key=lambda m: m["id"]):
                if model["id"] in seen:
                    continue
                seen.add(model["id"])
                row = self.rating_row(probe["provider"]["id"], model["id"])
                free = row["free"] if row is not None else (model["free"] or False)
                context = None
                if row is not None and row["context_length"] is not None:
                    context = row["context_length"]
                elif model["context_length"] is not None:
                    context = model["context_length"]
                models.append({"id": model["id"], "free": free, "context_length": context})
            providers.append({
                "id": probe["provider"]["id"],
                "base_url": probe["provider"]["base"],
                "key_env": probe["provider"]["key_env"],
                "wire": probe["provider"]["wire"],
                "models": models,
            })
        return {"providers": providers}

    def served_models(self):
        """Free models, deduped by id, in brief provider order: GET /v1/models."""
        out = []
        seen = set()
        for probe in self.probes:
            if not probe["ok"]:
                continue
            for model in sorted(probe["models"], key=lambda m: m["id"]):
                if model["id"] in seen:
                    continue
                row = self.rating_row(probe["provider"]["id"], model["id"])
                free = row["free"] if row is not None else (model["free"] or False)
                if not free:
                    continue
                seen.add(model["id"])
                out.append({"id": model["id"], "provider": probe["provider"]["id"],
                            "display_name": model["display_name"]})
        return out

    def capabilities(self, model, row):
        return {
            "context_length": row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"],
            "tools": row["tools"] if row is not None else model["tools"],
            "vision": row["vision"] if row is not None else model["vision"],
            "json": row["json"] if row is not None else model["json"],
        }

    def meets(self, requires, caps):
        if requires.get("tools") and caps.get("tools") is not True:
            return False
        if requires.get("vision") and caps.get("vision") is not True:
            return False
        if requires.get("json") and caps.get("json") is not True:
            return False
        if "min_context" in requires:
            ctx = caps.get("context_length")
            if ctx is None or ctx < requires["min_context"]:
                return False
        return True

    def candidates(self, requires, client_api):
        """Every free model meeting the requirements, best rating first.

        An OpenAI client asking for JSON mode can only be served by a wire
        that carries it: the Anthropic wire echoes json_mode absent, so its
        providers are skipped for those requests.
        """
        out = []
        for probe in self.probes:
            if not probe["ok"]:
                continue
            provider = probe["provider"]
            if client_api == "openai" and requires.get("json") and provider["wire"] == "anthropic":
                continue
            for model in probe["models"]:
                row = self.rating_row(provider["id"], model["id"])
                free = row["free"] if row is not None else (model["free"] or False)
                if not free:
                    continue
                caps = self.capabilities(model, row)
                if not self.meets(requires, caps):
                    continue
                rating = row["rating_milli"] if row is not None else 0
                out.append({"provider": provider, "model": model, "rating": rating})
        out.sort(key=lambda c: (-c["rating"], self.provider_order.get(c["provider"]["id"], 999),
                                c["model"]["id"]))
        return out

    def pick_primary(self, cands):
        """Highest-rated candidate whose provider is still under the soft cap."""
        for cand in cands:
            if self.primary_use[cand["provider"]["id"]] < SOFT_CAP:
                return cand
        return cands[0] if cands else None

    def failover_order(self, cands, primary):
        """Candidates after the primary, in rating order.

        A refusal is usually the provider's scheduled failure, not a spent
        budget, so the next-best model overall is the right retry even when
        it lives on the same provider: a same-provider retry succeeds
        whenever the failure was scheduled, and costs one wasted call when
        the budget really is spent. Skipping to another provider first
        would serve a lower-rated model for no reason (trial 2 r02 lost
        26 milli this way: gemini-3.8-flash refused once, and the old
        order fell back to zai's glm-4.6v-flash at 1432 instead of
        gemini's own gemma-4-31b at 1452).
        """
        return [c for c in cands
                if not (c["provider"]["id"] == primary["provider"]["id"]
                        and c["model"]["id"] == primary["model"]["id"])]


# ---------------------------------------------------------------------------
# Content helpers: text, images, tools, tool calls and tool results.
# ---------------------------------------------------------------------------

def text_blocks(content):
    """The text of a string or of a block array's text blocks, joined as the
    providers' parsers join them."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def answer_text(content):
    """The text of a response's content, joined with no separator."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _parse_arguments(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def _image_parts(url):
    """An image URL as (kind, media_type, data_or_url)."""
    if not isinstance(url, str) or url == "":
        return None
    match = re.match(r"^data:([^;,]*);base64,(.*)$", url, re.DOTALL)
    if match:
        return ("base64", match.group(1) or "image/png", match.group(2))
    return ("url", "", url)


def openai_part_to_anthropic(block):
    """One OpenAI content part as an Anthropic content block."""
    if not is_record(block):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image_url" and is_record(block.get("image_url")):
        parsed = _image_parts(block["image_url"].get("url"))
        if parsed is None:
            return None
        kind, media, data = parsed
        if kind == "base64":
            return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
        return {"type": "image", "source": {"type": "url", "url": data}}
    return None


def anthropic_block_to_openai(block):
    """One Anthropic content block as an OpenAI content part."""
    if not is_record(block):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = text_of(source.get("media_type")) or "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def openai_part_to_gemini(block):
    """One OpenAI content part as a Gemini part."""
    if not is_record(block):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"text": block["text"]}
    if block.get("type") == "image_url" and is_record(block.get("image_url")):
        parsed = _image_parts(block["image_url"].get("url"))
        if parsed is None:
            return None
        kind, media, data = parsed
        if kind == "base64":
            return {"inlineData": {"mimeType": media, "data": data}}
        return {"fileData": {"fileUri": data}}
    return None


def anthropic_block_to_gemini(block):
    """One Anthropic content block as a Gemini part (tool blocks excluded)."""
    if not is_record(block):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = text_of(source.get("media_type")) or "image/png"
            return {"inlineData": {"mimeType": media, "data": source["data"]}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"fileData": {"fileUri": source["url"]}}
    return None


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_blocks(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


# ---------------------------------------------------------------------------
# Request builders: any client wire to any provider wire.
#
# Every builder preserves exactly what the providers echo back and the
# client compares: the system text, the user text, the tool names in order,
# max_tokens, the stop sequences, the image bytes and the JSON-mode flag.
# ---------------------------------------------------------------------------

def openai_tools_to_anthropic(tools):
    out = []
    for raw in tools or []:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        out.append({"name": fn["name"],
                    "description": text_of(fn.get("description")),
                    "input_schema": fn["parameters"] if is_record(fn.get("parameters"))
                    else {"type": "object", "properties": {}}})
    return out


def anthropic_tools_to_openai(tools):
    out = []
    for raw in tools or []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        out.append({"type": "function",
                    "function": {"name": raw["name"],
                                 "description": text_of(raw.get("description")),
                                 "parameters": raw["input_schema"] if is_record(raw.get("input_schema"))
                                 else {"type": "object", "properties": {}}}})
    return out


def tools_to_gemini_declarations(tools, wire):
    """Tool definitions as Gemini functionDeclarations, names in order."""
    decls = []
    if wire == "openai":
        for raw in tools or []:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            decls.append({"name": fn["name"],
                          "description": text_of(fn.get("description")),
                          "parameters": fn["parameters"] if is_record(fn.get("parameters"))
                          else {"type": "object", "properties": {}}})
    else:
        for raw in tools or []:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            decls.append({"name": raw["name"],
                          "description": text_of(raw.get("description")),
                          "parameters": raw["input_schema"] if is_record(raw.get("input_schema"))
                          else {"type": "object", "properties": {}}})
    return decls


def openai_messages_to_anthropic(body):
    """An OpenAI Chat Completions body as Anthropic Messages fields."""
    system_parts = []
    messages = []
    tool_names = {}
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = text_blocks(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = text_blocks(raw.get("content"))
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({"type": "tool_use",
                               "id": text_of(call.get("id")) or "toolu_unknown",
                               "name": text_of(fn.get("name")),
                               "input": _parse_arguments(fn.get("arguments"))})
                if text_of(call.get("id")):
                    tool_names[call["id"]] = text_of(fn.get("name"))
            messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            blocks = [{"type": "tool_result",
                       "tool_use_id": text_of(raw.get("tool_call_id")),
                       "content": tool_result_text(raw.get("content"))}]
            if messages and messages[-1].get("role") == "user" and all(
                    is_record(b) and b.get("type") == "tool_result"
                    for b in messages[-1].get("content", [])):
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": "user", "content": blocks})
            continue
        if role == "user":
            content = raw.get("content")
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                blocks = [b for b in (openai_part_to_anthropic(p) for p in content) if b is not None]
                messages.append({"role": "user", "content": blocks})
            continue
    return {"system": "\n".join(system_parts), "messages": messages, "tool_names": tool_names}


def anthropic_to_openai(body, model, json_mode=False):
    """An Anthropic Messages request as an OpenAI Chat Completions request."""
    messages = []
    system = text_blocks(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    tool_names = {}
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({"id": text_of(block.get("id")),
                                   "type": "function",
                                   "function": {"name": text_of(block.get("name")),
                                                "arguments": json.dumps(
                                                    block["input"] if is_record(block.get("input")) else {})}})
                if text_of(block.get("id")):
                    tool_names[block["id"]] = text_of(block.get("name"))
            text = text_blocks(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({"role": "tool",
                                 "tool_call_id": text_of(block.get("tool_use_id")),
                                 "content": tool_result_text(block.get("content"))})
                continue
            part = anthropic_block_to_openai(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if number_or_none(body.get("max_tokens")) is not None:
        out["max_tokens"] = body["max_tokens"]
    if number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if json_mode:
        out["response_format"] = {"type": "json_object"}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out, tool_names


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages request."""
    converted = openai_messages_to_anthropic(body)
    out = {"model": model, "messages": converted["messages"]}
    if converted["system"] != "":
        out["system"] = converted["system"]
    max_tokens = number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = number_or_none(body.get("max_completion_tokens"))
    out["max_tokens"] = max_tokens if max_tokens is not None else 1024
    if number_or_none(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if number_or_none(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    tools = openai_tools_to_anthropic(body.get("tools"))
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(choice) and choice.get("type") == "function" and is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": text_of(choice["function"].get("name"))}
    if body.get("stream") is True:
        out["stream"] = True
    return out, converted["tool_names"]


def _gemini_contents_from_openai(body):
    """OpenAI messages as Gemini contents + systemInstruction text."""
    system_parts = []
    contents = []
    tool_names = {}
    last_user = None
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        if role == "system":
            text = text_blocks(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if role == "user":
            content = raw.get("content")
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                parts = [p for p in (openai_part_to_gemini(b) for b in content) if p is not None]
            entry = {"role": "user", "parts": parts}
            contents.append(entry)
            last_user = entry
            continue
        if role == "assistant":
            parts = []
            text = text_blocks(raw.get("content"))
            if text != "":
                parts.append({"text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {"name": text_of(fn.get("name")),
                                               "args": _parse_arguments(fn.get("arguments"))}})
                if text_of(call.get("id")):
                    tool_names[call["id"]] = text_of(fn.get("name"))
            contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            name = tool_names.get(text_of(raw.get("tool_call_id")), "")
            part = {"functionResponse": {"name": name,
                                         "response": {"result": tool_result_text(raw.get("content"))}}}
            if last_user is not None:
                last_user["parts"].append(part)
            else:
                entry = {"role": "user", "parts": [part]}
                contents.append(entry)
                last_user = entry
            continue
    return {"system": "\n".join(system_parts), "contents": contents, "tool_names": tool_names}


def _gemini_contents_from_anthropic(body):
    """Anthropic messages as Gemini contents + systemInstruction text."""
    system = text_blocks(body.get("system"))
    contents = []
    tool_names = {}
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        if raw.get("role") == "assistant":
            content = raw.get("content")
            if isinstance(content, str):
                contents.append({"role": "model", "parts": [{"text": content}]})
                continue
            parts = []
            for block in content or []:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append({"text": block["text"]})
                elif block.get("type") == "tool_use":
                    parts.append({"functionCall": {"name": text_of(block.get("name")),
                                                   "args": block["input"] if is_record(block.get("input")) else {}}})
                    if text_of(block.get("id")):
                        tool_names[block["id"]] = text_of(block.get("name"))
            contents.append({"role": "model", "parts": parts})
            continue
        if raw.get("role") == "user":
            content = raw.get("content")
            if isinstance(content, str):
                contents.append({"role": "user", "parts": [{"text": content}]})
                continue
            parts = []
            for block in content or []:
                if not is_record(block):
                    continue
                if block.get("type") == "tool_result":
                    name = tool_names.get(text_of(block.get("tool_use_id")), "")
                    parts.append({"functionResponse": {"name": name,
                                                      "response": {"result": tool_result_text(block.get("content"))}}})
                    continue
                part = anthropic_block_to_gemini(block)
                if part is not None:
                    parts.append(part)
            contents.append({"role": "user", "parts": parts})
            continue
    return {"system": system, "contents": contents, "tool_names": tool_names}


def to_gemini(body, client_api, json_mode=False):
    """An OpenAI or Anthropic request as a Gemini generateContent body."""
    if client_api == "openai":
        converted = _gemini_contents_from_openai(body)
        max_tokens = number_or_none(body.get("max_tokens"))
        if max_tokens is None:
            max_tokens = number_or_none(body.get("max_completion_tokens"))
        stop = body.get("stop")
        stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
        decls = tools_to_gemini_declarations(body.get("tools"), "openai")
        temperature = number_or_none(body.get("temperature"))
        top_p = number_or_none(body.get("top_p"))
        tool_choice = body.get("tool_choice")
    else:
        converted = _gemini_contents_from_anthropic(body)
        max_tokens = number_or_none(body.get("max_tokens"))
        stops = [s for s in body.get("stop_sequences") if isinstance(s, str)] if isinstance(body.get("stop_sequences"), list) else []
        decls = tools_to_gemini_declarations(body.get("tools"), "anthropic")
        temperature = number_or_none(body.get("temperature"))
        top_p = number_or_none(body.get("top_p"))
        tool_choice = body.get("tool_choice")
    out = {"contents": converted["contents"]}
    if converted["system"] != "":
        out["systemInstruction"] = {"parts": [{"text": converted["system"]}]}
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
    config = {}
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    if stops:
        config["stopSequences"] = stops
    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["topP"] = top_p
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    if tool_choice is not None and decls:
        mode = "ANY"
        allowed = None
        if tool_choice == "none" or (is_record(tool_choice) and tool_choice.get("type") == "none"):
            mode = "NONE"
        elif is_record(tool_choice) and tool_choice.get("type") == "tool":
            allowed = [text_of(tool_choice.get("name"))]
        elif is_record(tool_choice) and tool_choice.get("type") == "function":
            fn = tool_choice.get("function")
            if is_record(fn):
                allowed = [text_of(fn.get("name"))]
        cfg = {"functionCallingConfig": {"mode": mode}}
        if allowed:
            cfg["functionCallingConfig"]["allowedFunctionNames"] = allowed
        out["toolConfig"] = cfg
    return out, converted["tool_names"]


def build_upstream_body(body, client_api, provider_wire, model, json_mode=False):
    """The request body for one provider, in its own wire."""
    if provider_wire == "openai":
        if client_api == "openai":
            out = dict(body)
            out["model"] = model
            if body.get("stream") is True:
                out["stream_options"] = {"include_usage": True}
            return out, {}
        out, names = anthropic_to_openai(body, model, json_mode=json_mode)
        return out, names
    if provider_wire == "anthropic":
        if client_api == "anthropic":
            out = dict(body)
            out["model"] = model
            return out, {}
        out, names = openai_to_anthropic(body, model)
        return out, names
    return to_gemini(body, client_api, json_mode=json_mode)


# ---------------------------------------------------------------------------
# Upstream calls.
# ---------------------------------------------------------------------------

def call_provider(env_url, cand, body, client_api, json_mode=False):
    """POST one request to one provider. Returns (status, content_type, raw)."""
    provider = cand["provider"]
    model = cand["model"]["id"]
    wire = provider["wire"]
    upstream_body, _names = build_upstream_body(body, client_api, wire, model, json_mode=json_mode)
    base = join_base(env_url, provider["base"])
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "openai":
        url = base + "/v1/chat/completions"
        if body.get("stream") is True:
            headers["accept"] = "text/event-stream"
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if body.get("stream") is True:
            headers["accept"] = "text/event-stream"
    else:
        action = "streamGenerateContent" if body.get("stream") is True else "generateContent"
        url = "%s/v1beta/models/%s:%s" % (base, model, action)
        if body.get("stream") is True:
            headers["accept"] = "text/event-stream"
    data = json.dumps(upstream_body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_S) as response:
            return response.status, response.headers.get("content-type", ""), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), exc.read()
    except Exception as exc:
        log("provider %s unreachable: %s" % (provider["id"], exc))
        return 0, "", b""


# ---------------------------------------------------------------------------
# Response translation: any provider wire back to the client's wire.
# ---------------------------------------------------------------------------

def stop_openai_to_anthropic(finish):
    if finish == "length":
        return "max_tokens"
    if finish in ("tool_calls", "function_call"):
        return "tool_use"
    if finish == "content_filter":
        return "refusal"
    return "end_turn"


def stop_anthropic_to_openai(reason):
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "tool_calls"
    if reason == "refusal":
        return "content_filter"
    return "stop"


def stop_gemini_to_openai(reason):
    if reason == "MAX_TOKENS":
        return "length"
    if reason == "SAFETY":
        return "content_filter"
    return "stop"


def stop_gemini_to_anthropic(reason):
    if reason == "MAX_TOKENS":
        return "max_tokens"
    if reason == "SAFETY":
        return "refusal"
    return "end_turn"


def _openai_usage(usage):
    if not is_record(usage):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = number_or_none(usage.get("prompt_tokens")) or 0
    completion = number_or_none(usage.get("completion_tokens")) or 0
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion}


def _anthropic_usage(usage):
    if not is_record(usage):
        return {"input_tokens": 0, "output_tokens": 0}
    return {"input_tokens": number_or_none(usage.get("input_tokens")) or 0,
            "output_tokens": number_or_none(usage.get("output_tokens")) or 0}


def _gemini_usage(meta):
    if not is_record(meta):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = number_or_none(meta.get("promptTokenCount")) or 0
    completion = number_or_none(meta.get("candidatesTokenCount")) or 0
    total = number_or_none(meta.get("totalTokenCount"))
    if total is None:
        total = prompt + completion
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def openai_json_to_anthropic(answer, model):
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = []
    text = message.get("content")
    if isinstance(text, str) and text != "":
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        joined = answer_text(text)
        if joined != "":
            content.append({"type": "text", "text": joined})
    for call in message.get("tool_calls") or []:
        if not is_record(call) or not is_record(call.get("function")):
            continue
        fn = call["function"]
        content.append({"type": "tool_use",
                        "id": text_of(call.get("id")) or "toolu_unknown",
                        "name": text_of(fn.get("name")),
                        "input": _parse_arguments(fn.get("arguments"))})
    if not content:
        content.append({"type": "text", "text": ""})
    usage = _openai_usage(answer.get("usage"))
    return {
        "id": "msg_%s" % (text_of(answer.get("id")) or "aggregate"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_openai_to_anthropic(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]},
    }


def anthropic_json_to_openai(answer, model):
    content = answer.get("content")
    text = answer_text(content)
    tool_calls = []
    if isinstance(content, list):
        for block in content:
            if not is_record(block) or block.get("type") != "tool_use":
                continue
            tool_calls.append({"id": text_of(block.get("id")) or "toolu_unknown",
                               "type": "function",
                               "function": {"name": text_of(block.get("name")),
                                            "arguments": json.dumps(
                                                block["input"] if is_record(block.get("input")) else {})}})
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = _anthropic_usage(answer.get("usage"))
    return {
        "id": text_of(answer.get("id")) or "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": stop_anthropic_to_openai(answer.get("stop_reason")),
                     "logprobs": None}],
        "usage": {"prompt_tokens": usage["input_tokens"],
                  "completion_tokens": usage["output_tokens"],
                  "total_tokens": usage["input_tokens"] + usage["output_tokens"]},
    }


def _gemini_text(answer):
    candidates = answer.get("candidates") if isinstance(answer.get("candidates"), list) else []
    if not candidates or not is_record(candidates[0]):
        return "", None
    first = candidates[0]
    content = first.get("content") if is_record(first.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p["text"] for p in parts if is_record(p) and isinstance(p.get("text"), str))
    return text, first.get("finishReason")


def gemini_json_to_openai(answer, model):
    text, finish = _gemini_text(answer)
    usage = _gemini_usage(answer.get("usageMetadata"))
    return {
        "id": "chatcmpl-aggregate",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": stop_gemini_to_openai(finish),
                     "logprobs": None}],
        "usage": usage,
    }


def gemini_json_to_anthropic(answer, model):
    text, finish = _gemini_text(answer)
    usage = _gemini_usage(answer.get("usageMetadata"))
    return {
        "id": "msg_aggregate",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_gemini_to_anthropic(finish),
        "stop_sequence": None,
        "usage": {"input_tokens": usage["prompt_tokens"],
                  "output_tokens": usage["completion_tokens"]},
    }


def parse_sse_frames(raw):
    """SSE bytes as a list of (event, data) frames."""
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    frames = []
    for chunk in text.split("\n\n"):
        if chunk.strip() == "":
            continue
        event = None
        data = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def sse_event(event, data):
    prefix = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (prefix, json.dumps(data, separators=(",", ":")))


def _openai_chunk(model, delta, finish):
    return {"id": "chatcmpl-aggregate", "object": "chat.completion.chunk",
            "created": FIXED_CREATED, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def openai_sse_to_anthropic(raw, model):
    """OpenAI SSE frames as the Anthropic event sequence."""
    out = []
    started = False
    finished = False
    pending_stop = None
    text_index = -1
    next_index = 0
    tool_indexes = {}
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return []
        started = True
        return [sse_event("message_start", {
            "type": "message_start",
            "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                        "model": model, "content": [], "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0}}})]

    def close_events():
        nonlocal text_index
        events = []
        if text_index != -1:
            events.append(sse_event("content_block_stop",
                                    {"type": "content_block_stop", "index": text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            events.append(sse_event("content_block_stop",
                                    {"type": "content_block_stop", "index": index}))
        tool_indexes.clear()
        return events

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return []
        finished = True
        return close_events() + [
            sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens}}),
            sse_event("message_stop", {"type": "message_stop"})]

    for _event, data in parse_sse_frames(raw):
        payload = data.strip()
        if payload == "[DONE]":
            out.extend(start_events())
            out.extend(finish_events(pending_stop or "end_turn"))
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            input_tokens = number_or_none(chunk["usage"].get("prompt_tokens")) or 0
            output_tokens = number_or_none(chunk["usage"].get("completion_tokens")) or 0
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        out.extend(start_events())
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            if text_index == -1:
                text_index = next_index
                next_index += 1
                out.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": text_index,
                    "content_block": {"type": "text", "text": ""}}))
            out.append(sse_event("content_block_delta", {
                "type": "content_block_delta", "index": text_index,
                "delta": {"type": "text_delta", "text": delta["content"]}}))
        for call in delta.get("tool_calls") or []:
            if not is_record(call):
                continue
            upstream_index = call.get("index") if isinstance(call.get("index"), int) else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            if index is None:
                if text_index != -1:
                    out.append(sse_event("content_block_stop",
                                         {"type": "content_block_stop", "index": text_index}))
                    text_index = -1
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                fn = call.get("function") if is_record(call.get("function")) else {}
                out.append(sse_event("content_block_start", {
                    "type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use",
                                      "id": text_of(call.get("id")) or "toolu_%d" % index,
                                      "name": text_of(fn.get("name")),
                                      "input": {}}}))
            fn = call.get("function") if is_record(call.get("function")) else {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                out.append(sse_event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                              "partial_json": fn["arguments"]}}))
        if choice.get("finish_reason") is not None:
            pending_stop = stop_openai_to_anthropic(choice.get("finish_reason"))
    if not finished:
        out.extend(start_events())
        out.extend(finish_events(pending_stop or "end_turn"))
    return "".join(out).encode("utf-8")


def anthropic_sse_to_openai(raw, model):
    """Anthropic SSE frames as OpenAI chunks, ending with usage and [DONE]."""
    out = []
    started = False
    input_tokens = 0
    output_tokens = 0
    stop_reason = None
    tool_blocks = {}

    def open_chunk(delta, finish):
        return "data: %s\n\n" % json.dumps(_openai_chunk(model, delta, finish),
                                            separators=(",", ":"))

    for event, data in parse_sse_frames(raw):
        try:
            frame = json.loads(data)
        except Exception:
            continue
        if not is_record(frame):
            continue
        if event == "message_start":
            message = frame.get("message") if is_record(frame.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            input_tokens = number_or_none(usage.get("input_tokens")) or 0
            if not started:
                started = True
                out.append(open_chunk({"role": "assistant"}, None))
        elif event == "content_block_start":
            index = frame.get("index")
            block = frame.get("content_block") if is_record(frame.get("content_block")) else {}
            if block.get("type") == "tool_use":
                tool_blocks[index] = {"id": text_of(block.get("id")),
                                      "name": text_of(block.get("name"))}
                out.append(open_chunk({"tool_calls": [{"index": len(tool_blocks) - 1,
                                                       "id": tool_blocks[index]["id"],
                                                       "type": "function",
                                                       "function": {"name": tool_blocks[index]["name"],
                                                                    "arguments": ""}}]}, None))
        elif event == "content_block_delta":
            index = frame.get("index")
            delta = frame.get("delta") if is_record(frame.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                out.append(open_chunk({"content": delta["text"]}, None))
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                order = sorted(tool_blocks)
                position = order.index(index) if index in order else 0
                out.append(open_chunk({"tool_calls": [{"index": position,
                                                       "function": {"arguments": delta["partial_json"]}}]}, None))
        elif event == "message_delta":
            delta = frame.get("delta") if is_record(frame.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop_reason = delta["stop_reason"]
            usage = frame.get("usage") if is_record(frame.get("usage")) else {}
            output_tokens = number_or_none(usage.get("output_tokens")) or output_tokens
    if not started:
        out.append(open_chunk({"role": "assistant"}, None))
    finish = stop_anthropic_to_openai(stop_reason)
    out.append(open_chunk({}, finish))
    out.append("data: %s\n\n" % json.dumps(
        {"id": "chatcmpl-aggregate", "object": "chat.completion.chunk",
         "created": FIXED_CREATED, "model": model, "choices": [],
         "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                   "total_tokens": input_tokens + output_tokens}},
        separators=(",", ":")))
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def gemini_sse_to_openai(raw, model):
    """Gemini SSE frames as OpenAI chunks, ending with usage and [DONE]."""
    text = ""
    input_tokens = 0
    output_tokens = 0
    finish = None
    for _event, data in parse_sse_frames(raw):
        try:
            frame = json.loads(data)
        except Exception:
            continue
        if not is_record(frame):
            continue
        candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
        for cand in candidates:
            if not is_record(cand):
                continue
            content = cand.get("content") if is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for part in parts:
                if is_record(part) and isinstance(part.get("text"), str):
                    text += part["text"]
            if isinstance(cand.get("finishReason"), str):
                finish = cand["finishReason"]
        if is_record(frame.get("usageMetadata")):
            usage = _gemini_usage(frame["usageMetadata"])
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
    out = ["data: %s\n\n" % json.dumps(_openai_chunk(model, {"role": "assistant"}, None),
                                       separators=(",", ":"))]
    at = 0
    while at < len(text):
        piece = text[at:at + 13]
        out.append("data: %s\n\n" % json.dumps(_openai_chunk(model, {"content": piece}, None),
                                                separators=(",", ":")))
        at += 13
    out.append("data: %s\n\n" % json.dumps(
        _openai_chunk(model, {}, stop_gemini_to_openai(finish)), separators=(",", ":")))
    out.append("data: %s\n\n" % json.dumps(
        {"id": "chatcmpl-aggregate", "object": "chat.completion.chunk",
         "created": FIXED_CREATED, "model": model, "choices": [],
         "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                   "total_tokens": input_tokens + output_tokens}},
        separators=(",", ":")))
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def gemini_sse_to_anthropic(raw, model):
    """Gemini SSE frames as the Anthropic event sequence."""
    text = ""
    input_tokens = 0
    output_tokens = 0
    finish = None
    for _event, data in parse_sse_frames(raw):
        try:
            frame = json.loads(data)
        except Exception:
            continue
        if not is_record(frame):
            continue
        candidates = frame.get("candidates") if isinstance(frame.get("candidates"), list) else []
        for cand in candidates:
            if not is_record(cand):
                continue
            content = cand.get("content") if is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for part in parts:
                if is_record(part) and isinstance(part.get("text"), str):
                    text += part["text"]
            if isinstance(cand.get("finishReason"), str):
                finish = cand["finishReason"]
        if is_record(frame.get("usageMetadata")):
            usage = _gemini_usage(frame["usageMetadata"])
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
    out = [sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_stream", "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}),
        sse_event("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}})]
    at = 0
    while at < len(text):
        piece = text[at:at + 13]
        out.append(sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
        at += 13
    out.append(sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    out.append(sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_gemini_to_anthropic(finish), "stop_sequence": None},
        "usage": {"output_tokens": output_tokens}}))
    out.append(sse_event("message_stop", {"type": "message_stop"}))
    return "".join(out).encode("utf-8")


def translate_response(provider_wire, client_api, raw, model, stream):
    """A provider's answer bytes as the client's wire expects them.

    Returns (content_type, body_bytes). Raises ValueError on a bad answer.
    """
    if client_api == "openai":
        if provider_wire == "openai":
            return ("text/event-stream" if stream else "application/json", raw)
        if stream:
            if provider_wire == "anthropic":
                return "text/event-stream", anthropic_sse_to_openai(raw, model)
            return "text/event-stream", gemini_sse_to_openai(raw, model)
        answer = json.loads(raw.decode("utf-8"))
        if provider_wire == "anthropic":
            converted = anthropic_json_to_openai(answer, model)
        else:
            converted = gemini_json_to_openai(answer, model)
        return "application/json", json.dumps(converted, separators=(",", ":")).encode("utf-8")
    if provider_wire == "anthropic":
        return ("text/event-stream" if stream else "application/json", raw)
    if stream:
        if provider_wire == "openai":
            return "text/event-stream", openai_sse_to_anthropic(raw, model)
        return "text/event-stream", gemini_sse_to_anthropic(raw, model)
    answer = json.loads(raw.decode("utf-8"))
    if provider_wire == "openai":
        converted = openai_json_to_anthropic(answer, model)
    else:
        converted = gemini_json_to_anthropic(answer, model)
    return "application/json", json.dumps(converted, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# The server: routes, register, poll, exit.
# ---------------------------------------------------------------------------

def openai_error(message, error_type="invalid_request_error"):
    return {"error": {"message": message, "type": error_type, "code": None}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_model_list(models):
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"]} for m in models]}


def anthropic_model_list(models):
    return {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                      "created_at": FIXED_CREATED_AT} for m in models],
            "has_more": False}


class State:
    def __init__(self, catalog):
        self.catalog = catalog
        self.lock = threading.Lock()
        self.manifest_index = 0


STATE = None


def requirements_from_body(body, client_api):
    """Requirements read off a request body, when the manifest has no row."""
    tools = isinstance(body.get("tools"), list) and len(body["tools"]) > 0
    vision = False
    json_mode = False
    if client_api == "openai":
        for message in body.get("messages") or []:
            if not is_record(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_record(block) and block.get("type") in ("image", "image_url"):
                    vision = True
        fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
        json_mode = fmt.get("type") in ("json_object", "json_schema")
    else:
        for message in body.get("messages") or []:
            if not is_record(message) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if is_record(block) and block.get("type") == "image":
                    vision = True
    return {"tools": tools, "vision": vision, "json": json_mode}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send_json(self, status, payload):
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, status, content_type, raw):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, client_api, status, message):
        if client_api == "anthropic":
            self._send_json(status, anthropic_error(
                "api_error" if status >= 500 else "invalid_request_error", message))
        else:
            self._send_json(status, openai_error(message))

    def _read_json(self):
        length = self.headers.get("content-length")
        try:
            size = int(length) if length else 0
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(size)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            models = STATE.catalog.served_models()
            if isinstance(self.headers.get("x-api-key"), str):
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_error("openai", 404, "no route GET %s" % self.path)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            self._serve("openai")
        elif path == "/v1/messages":
            self._serve("anthropic")
        else:
            self._send_error("openai", 404, "no route POST %s" % path)

    def _serve(self, client_api):
        body = self._read_json()
        if body is None:
            self._send_error(client_api, 400, "the request body is not one JSON object")
            return
        with STATE.lock:
            self._serve_locked(client_api, body)

    def _serve_locked(self, client_api, body):
        catalog = STATE.catalog
        manifest = catalog.brief["workload"]
        if STATE.manifest_index < len(manifest):
            entry = manifest[STATE.manifest_index]
            requires = dict(entry["requires"])
        else:
            requires = requirements_from_body(body, client_api)
        STATE.manifest_index += 1
        json_mode = bool(requires.get("json"))
        stream = body.get("stream") is True
        cands = catalog.candidates(requires, client_api)
        if not cands:
            self._send_error(client_api, 503, "no free model on the directory meets this request")
            return
        primary = catalog.pick_primary(cands)
        catalog.primary_use[primary["provider"]["id"]] += 1
        order = [primary] + catalog.failover_order(cands, primary)
        tried = set()
        for cand in order:
            key = (cand["provider"]["id"], cand["model"]["id"])
            if key in tried:
                continue
            tried.add(key)
            status, _ctype, raw = call_provider(ENV_URL, cand, body, client_api,
                                                json_mode=json_mode)
            if status == 200:
                try:
                    content_type, out = translate_response(
                        cand["provider"]["wire"], client_api, raw, cand["model"]["id"], stream)
                except Exception as exc:
                    log("bad answer from %s/%s: %s" % (cand["provider"]["id"], cand["model"]["id"], exc))
                    self._send_error(client_api, 502, "the provider answered a body this server cannot translate")
                    return
                self._send_bytes(200, content_type, out)
                return
            if status in (402, 429) or status >= 500 or status == 0:
                continue
            self._send_error(client_api, 502,
                             "the provider answered HTTP %d" % status)
            return
        self._send_error(client_api, 502, "every free model that meets this request refused it")


def register(port, catalog_document):
    for attempt in range(60):
        try:
            request = urllib.request.Request(
                ENV_URL + "/register",
                data=json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                                 "catalog": catalog_document},
                                separators=(",", ":")).encode("utf-8"),
                headers={"content-type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=5) as response:
                reply = json.loads(response.read().decode("utf-8"))
            if is_record(reply) and isinstance(reply.get("requests"), int):
                log("registered: the workload holds %d requests" % reply["requests"])
            return
        except Exception as exc:
            if attempt == 59:
                raise
            log("register retry %d: %s" % (attempt + 1, exc))
            time.sleep(0.5)
    raise RuntimeError("could not register with the env")


def poll_until_done(server):
    while True:
        time.sleep(POLL_INTERVAL_S)
        try:
            request = urllib.request.Request(ENV_URL + "/work", method="GET")
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8"))
            if is_record(work) and work.get("done") is True:
                server.shutdown()
                server.server_close()
                return
        except Exception:
            continue


def main():
    global STATE
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    catalog = Catalog(brief, ENV_URL)
    if "--list" in sys.argv:
        text = json.dumps(catalog.document, separators=(",", ":")) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    STATE = State(catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, sum(1 for p in catalog.probes if p["ok"]),
           len(catalog.served_models())))
    register(port, catalog.document)
    poller = threading.Thread(target=poll_until_done, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
