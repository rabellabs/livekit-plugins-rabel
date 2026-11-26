"""Tests for livekit-plugins-rabel: the wrapper's rules on a scripted speech-to-text against a local stand-in of the
Rabel API (versions 1, 2 and 3 of its conversation), and that it fails open (API down, slow, refusing, erroring: the events are
exactly the inner STT's). No test calls a live service."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import LanguageCode, __version__ as agents_version, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, TimedString
from livekit.plugins import rabel
from livekit.plugins.rabel.stt import words_of

pytestmark = pytest.mark.unit

T = stt.SpeechEventType


# ------------------------------------------------------------------ a scripted inner speech-to-text


class ScriptedSTT(stt.STT):
    """Emits (seconds after the first audio frame, event type, text, last word's confidence) in order."""

    def __init__(self, script, language="en"):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.script = script
        self.language = language

    @property
    def provider(self) -> str:
        return "Scripted"

    @property
    def model(self) -> str:
        return "Script-1"

    async def _recognize_impl(
        self, buffer, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS
    ):
        raise NotImplementedError

    def stream(self, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return ScriptedStream(stt=self, conn_options=conn_options)


class ScriptedStream(stt.RecognizeStream):
    async def _run(self) -> None:
        await self._input_ch.recv()
        t0 = time.monotonic()
        for at, typ, text, conf in self._stt.script:
            if typ == COMMIT:
                continue  # LiveKit's, not the speech-to-text's (collect() hands it to the wrapper)
            wait = t0 + at - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            alts = []
            if text is not None:
                words = [TimedString(w, confidence=0.99) for w in text.split()]
                if words and conf is not None:
                    words[-1] = TimedString(str(words[-1]), confidence=conf)
                alts = [
                    stt.SpeechData(
                        language=LanguageCode(getattr(self._stt, "language", "en")),
                        text=text,
                        words=words,
                        confidence=0.9,
                    )
                ]
            self._event_ch.send_nowait(stt.SpeechEvent(type=typ, alternatives=alts))
        async for _ in self._input_ch:
            pass


def ev(at, typ, text=None, conf=None):
    return (at, typ, text, conf)


# in a script: LiveKit commits a turn, which the wrapper sees as the caller's message LiveKit adds to the conversation
COMMIT = "livekit_commit"


def user_item(text):
    from livekit.agents import llm
    from livekit.agents.voice.events import ConversationItemAddedEvent

    return ConversationItemAddedEvent(item=llm.ChatMessage(role="user", content=[text or ""]))


# ------------------------------------------------------------------ a local stand-in of the Rabel API


class FakeAPI:
    """v: 1 answers as the API served before version 2 (a bare `ready`, guesses without `chance`); 2 answers as the
    API's version 2 (`ready` with `min_chance` and `hold_line`, guesses with `chance`): today's production, which reads
    the hookup's version 3 hello as a version 2 hello; 3 answers as an API that serves the new Bell (`ready` with `"v": 3`
    and `min_chance` as the hello set it, null when it set none, guesses with `chance`). The version 3 stand-in sends
    `hold_always` and `hold_updates` only through `ready_extra`, so each rule is tested alone; one test gives it the
    API's own values.
    mode: "ok" (guess `answers[heard words]` after `delay` s), "refuse" (an error message at once), "garbage" (answers
    with broken JSON, a guess for words never sent and a bad_message). `plan` gives a mode for each conversation in
    turn, the last repeating. `ready_after` delays the ready; `close_after` closes a conversation cleanly that long
    after its ready. `ready_extra` adds fields to a version 2 or 3 `ready` (a version 2 `ready` with rule 3's
    `hold_always` and `hold_updates`: fields this client takes from a version 3 `ready` only).
    `refuse_report` answers an `end_of_turn` that carries the report with 400 bad_message and closes."""

    def __init__(
        self,
        mode="ok",
        answers=None,
        delay=0.02,
        status=402,
        error="no_seconds",
        retry_after=60,
        ready_after=0.0,
        v=1,
        hold_line=0.9,
        chance=0.86,
        send_chance=True,
        min_chance=None,
        plan=None,
        close_after=None,
        ready_extra=None,
        refuse_report=False,
    ):
        self.mode, self.answers, self.delay = mode, answers or {}, delay
        self.status, self.error, self.retry_after = status, error, retry_after
        self.ready_after, self.v, self.hold_line = ready_after, v, hold_line
        self.chance, self.send_chance, self.min_chance = chance, send_chance, min_chance
        self.plan, self.close_after = plan or [mode], close_after
        self.ready_extra = ready_extra or {}
        self.refuse_report = refuse_report
        self.received: list[dict] = []
        self.hello = None
        self.connected_at: list[float] = []
        self.app = web.Application()
        self.app.router.add_get("/v1/conversation", self.conversation)

    def ready(self) -> dict:
        if self.v == 1:
            return {"type": "ready"}
        h = self.hello or {}
        if self.v == 3:
            min_chance = self.min_chance if self.min_chance is not None else h.get("min_chance")
            if min_chance is not None:
                min_chance = min(max(min_chance, 0.67), 0.99)
            return {
                "type": "ready",
                "v": 3,
                "agent": h.get("agent", "default"),
                "min_chance": min_chance,
                "hold_line": self.hold_line,
                "learning": h.get("learn", True),
                "calls_heard": 12,
                **self.ready_extra,
            }
        min_chance = self.min_chance
        if min_chance is None:  # the API clamps the hello's min_chance and says which is in force
            min_chance = min(max(h.get("min_chance", 0.7), 0.67), 0.99)
        return {
            "type": "ready",
            "v": 2,
            "agent": h.get("agent", "default"),
            "min_chance": min_chance,
            "hold_line": self.hold_line,
            "learning": h.get("learn", True),
            "calls_heard": 143,
            **self.ready_extra,
        }

    def guess(self, text: str, heard: str) -> dict:
        g = {"type": "guess", "turn": 0, "id": 1, "text": text, "heard": heard}
        if self.v in (2, 3) and self.send_chance:
            g["chance"] = self.chance
        return g

    async def conversation(self, req):
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        self.connected_at.append(time.monotonic())
        mode = self.plan[min(len(self.connected_at), len(self.plan)) - 1]
        self.hello = json.loads((await ws.receive()).data)
        if mode == "refuse":
            await ws.send_str(
                json.dumps(
                    {
                        "type": "error",
                        "status": self.status,
                        "error": self.error,
                        "message": "refused",
                        "retry_after_s": self.retry_after,
                    }
                )
            )
            await ws.close()
            return ws
        if self.ready_after:
            await asyncio.sleep(self.ready_after)
        await ws.send_str(json.dumps(self.ready()))
        if self.close_after is not None:
            asyncio.get_running_loop().call_later(
                self.close_after, lambda: asyncio.ensure_future(ws.close())
            )
        async for msg in ws:
            m = json.loads(msg.data)
            self.received.append(m)
            if self.refuse_report and m.get("type") == "end_of_turn" and "guess" in m:
                # an API that refuses the report's field (none does: an API that does not know it ignores it)
                await ws.send_str(
                    json.dumps(
                        {"type": "error", "status": 400, "error": "bad_message", "message": "x"}
                    )
                )
                await ws.close()
                return ws
            if mode == "garbage":
                await ws.send_str("{not json")
                await ws.send_str(json.dumps(self.guess("words never heard", "nothing")))
                await ws.send_str(
                    json.dumps(
                        {"type": "error", "status": 400, "error": "bad_message", "message": "x"}
                    )
                )
            if m.get("type") == "heard" and m["text"] in self.answers:
                g = self.guess(self.answers[m["text"]], m["text"])
                asyncio.get_running_loop().call_later(
                    self.delay, lambda g=g: asyncio.ensure_future(ws.send_str(json.dumps(g)))
                )
        return ws


async def serve(api: FakeAPI):
    runner = web.AppRunner(api.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def collect(s: stt.STT, settle=0.3, wait_ready: rabel.STT | None = None):
    stream = s.stream()
    if wait_ready is not None:
        # start the scripted words only once the wrapper's API connection has had its chance
        await asyncio.sleep(0.2)
    stream.push_frame(
        rtc.AudioFrame(
            data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
        )
    )
    out = []

    async def read():
        async for e in stream:
            out.append((e.type, e.alternatives[0].text if e.alternatives else None))

    reader = asyncio.create_task(read())
    script = s.wrapped_stt.script if isinstance(s, rabel.STT) else s.script
    if isinstance(s, rabel.STT):
        loop = asyncio.get_running_loop()
        for at, typ, text, _ in script:
            if typ == COMMIT:
                loop.call_later(at, s._push_conversation_item, user_item(text))
    await asyncio.sleep(max(at for at, *_ in script) + settle)
    stream.end_input()
    await asyncio.wait_for(reader, 5)
    await stream.aclose()
    return out


TURN = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
    ev(0.80, T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.", 0.95),
    ev(0.95, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ev(0.95, T.END_OF_SPEECH),
]


async def wrapped_run(script, api, **kw):
    runner, url = await serve(api)
    try:
        async with aiohttp.ClientSession() as http:
            w = rabel.STT(
                ScriptedSTT(script), api_key="test", base_url=url, http_session=http, **kw
            )
            out = await collect(w, wait_ready=w)
            return out, next(iter(w._streams)).counts if w._streams else {}
    finally:
        await runner.cleanup()


async def plain_run(script):
    return await collect(ScriptedSTT(script))


def hello_v3(agent="default", min_chance=None, learn=True):
    """the hello the hookup sends: `"type": "hello"` and `"v": 3`; `min_chance` only when the developer set one"""
    h = {
        "key": "test",
        "type": "hello",
        "v": 3,
        "agent": agent,
        "host": "livekit",
        "host_version": agents_version,
        "hookup": rabel.__version__,
        "vendor": "scripted/script-1",
        "learn": learn,
    }
    if min_chance is not None:
        h["min_chance"] = min_chance
    return h


# ------------------------------------------------------------------ the hello and the options


async def test_words_match_livekit_comparison():
    from livekit.agents.voice.agent_activity import _transcripts_equivalent

    for a, b in [
        ("Hello, how are you?", "hello how are you"),
        ("rush-hour", "rushhour"),
        ("can't", "cant"),
        ("I'd like", "id like"),
        ("a b", "a b c"),
    ]:
        assert (words_of(a) == words_of(b) and bool(words_of(a))) == _transcripts_equivalent(a, b)


@pytest.mark.parametrize("v", [2, 3], ids=["api-v2", "api-v3"])
async def test_hello_announces_version_3_and_sends_min_chance_only_when_set(v):
    # the same hello to today's production (version 2) and to an API that serves the new Bell
    api = FakeAPI(v=v)
    runner, url = await serve(api)
    try:
        async with aiohttp.ClientSession() as http:
            w = rabel.STT(
                ScriptedSTT(TURN),
                api_key="test",
                base_url=url,
                http_session=http,
                agent="front-desk",
            )
            await collect(w, wait_ready=w)
            assert api.hello == hello_v3(agent="front-desk")
            # update_options takes effect from the next speech stream
            w.update_options(agent="demo-bank", min_chance=0.9, learn=False)
            await collect(w, wait_ready=w)
            assert api.hello == hello_v3(agent="demo-bank", min_chance=0.9, learn=False)
            assert (
                rabel.STT(ScriptedSTT(TURN), api_key="test", learn=False)._hello()["learn"] is False
            )
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "kw",
    [
        {"agent": "Front Desk"},
        {"agent": ""},
        {"agent": "a" * 65},
        {"agent": "front.desk"},
        {"min_chance": True},
        {"min_chance": "0.8"},
        {"answer_timeout": 0},
        {"answer_timeout": -1.0},
        {"learn": "no"},
        {"learn": None},
    ],
    ids=lambda kw: "-".join(f"{k}={v!r}"[:24] for k, v in kw.items()),
)
async def test_options_are_checked(kw):
    with pytest.raises(ValueError):
        rabel.STT(ScriptedSTT(TURN), api_key="test", **kw)
    w = rabel.STT(ScriptedSTT(TURN), api_key="test", agent="demo-bank", min_chance=0.99)
    with pytest.raises(ValueError):
        w.update_options(**kw)
    assert w._agent == "demo-bank" and w._min_chance == 0.99


async def test_min_chance_left_out_is_not_sent_and_none_goes_back_to_it():
    w = rabel.STT(ScriptedSTT(TURN), api_key="test")
    assert w._min_chance is None and "min_chance" not in w._hello()
    w.update_options(min_chance=0.8)
    assert w._hello()["min_chance"] == 0.8
    w.update_options(min_chance=None)
    assert w._min_chance is None and "min_chance" not in w._hello()


@pytest.mark.parametrize("given, used", [(0.5, 0.67), (1.0, 0.99), (0.67, 0.67), (0.8, 0.8)])
async def test_min_chance_clamped_like_the_api(given, used, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    w = rabel.STT(ScriptedSTT(TURN), api_key="test", min_chance=given)
    assert w._min_chance == used and w._hello()["min_chance"] == used
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == (0 if given == used else 1), lines


# ------------------------------------------------------------------ the rules


@pytest.mark.parametrize("v", [1, 2], ids=["api-v1", "api-v2"])
async def test_guess_sent_once_beginnings_held_eager_equal_held(v):
    api = FakeAPI(answers={"I'd like to check": "i'd like to check my balance"}, v=v)
    out, counts = await wrapped_run(TURN, api)
    # the guess goes out as one preflight; the interim that is a beginning of it is held back (rule 2);
    # Flux's eager with the same words is held back (rule 5); the final and end of speech pass
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like"),
        (T.INTERIM_TRANSCRIPT, "I'd like to check"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance."),
        (T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
        (T.END_OF_SPEECH, None),
    ]
    assert (
        counts["guess_sent"] == 1
        and counts["interim_held_back"] == 1
        and counts["eager_held_back"] == 1
    )
    assert api.hello == hello_v3()
    heard = [m for m in api.received if m["type"] == "heard"]
    assert [m["text"] for m in heard] == [
        "I'd like",
        "I'd like to check",
        "I'd like to check my",
        "I'd like to check my balance.",
    ]
    # only the words: no word times, confidences, event kind or vendor
    assert all(set(m) == {"type", "text"} for m in heard), heard
    last = dict(api.received[-1])
    assert isinstance(last.pop("head_start_ms"), int)
    assert last == {"type": "end_of_turn", "text": "I'd like to check my balance.", "guess": "kept"}


GUESS_AT_CHECK = {"I'd like to check": "i'd like to check my balance"}


@pytest.mark.parametrize(
    "api_kw, stt_kw, dropped",
    [
        # under the min_chance in force: the ready's, which the API may have clamped
        pytest.param(
            {"chance": 0.75}, {"min_chance": 0.8}, "guess_below_min_chance", id="under-hello"
        ),
        pytest.param(
            {"chance": 0.75, "min_chance": 0.8}, {}, "guess_below_min_chance", id="under-ready"
        ),
        # a version 2 API must send a chance
        pytest.param({"send_chance": False}, {}, "guess_no_chance", id="no-chance"),
        pytest.param({"chance": 1.5}, {}, "guess_unreadable", id="chance-out-of-range"),
        # later than answer_timeout after its words
        pytest.param({"delay": 0.3}, {"answer_timeout": 0.1}, "guess_slow", id="slow"),
    ],
)
async def test_wrong_or_slow_guess_not_used(api_kw, stt_kw, dropped, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    api = FakeAPI(answers=GUESS_AT_CHECK, v=2, **api_kw)
    out, counts = await wrapped_run(TURN, api, **stt_kw)
    assert out == await plain_run(TURN)
    assert counts[dropped] == 1 and "guess_sent" not in counts
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == 1, lines


async def test_guess_at_min_chance_and_in_time_is_used():
    api = FakeAPI(answers=GUESS_AT_CHECK, v=2, chance=0.8, min_chance=0.8, delay=0.3)
    out, counts = await wrapped_run(TURN, api, answer_timeout=0.5)
    assert counts["guess_sent"] == 1
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.") in out


TWO_TURNS = [
    *TURN,
    ev(1.40, T.START_OF_SPEECH),
    ev(1.45, T.INTERIM_TRANSCRIPT, "and my", 0.95),
    ev(1.70, T.INTERIM_TRANSCRIPT, "and my savings", 0.95),
    ev(1.90, T.FINAL_TRANSCRIPT, "And my savings."),
    ev(1.90, T.END_OF_SPEECH),
]
# the same with LiveKit's commit of the first turn before the second begins (an early reply waits for that
# commit, and rule 8 holds a guess until it)
TWO_TURNS_COMMITTED = sorted(
    [*TWO_TURNS, ev(1.20, COMMIT, "I'd like to check my balance.")], key=lambda e: e[0]
)


@pytest.mark.parametrize(
    "v, script", [(2, TWO_TURNS), (3, TWO_TURNS_COMMITTED)], ids=["api-v2", "api-v3"]
)
async def test_conversation_opened_mid_turn_starts_with_the_next_turn(v, script):
    # ready comes about 0.3 s into the first turn: nothing of that turn is sent, the second turn is
    api = FakeAPI(answers={"and my": "and my savings please"}, v=v, ready_after=0.5)
    out, counts = await wrapped_run(script, api)
    sent = [(m["type"], m["text"]) for m in api.received]
    assert sent == [
        ("heard", "and my"),
        ("heard", "and my savings"),
        ("end_of_turn", "And my savings."),
    ]
    assert counts["guess_sent"] == 1 and (T.PREFLIGHT_TRANSCRIPT, "and my savings please.") in out


# ------------------------------------------------------------------ reopening (the API's docs)


async def test_reopen_waits():
    from livekit.plugins.rabel._bell import BellConnection

    b = BellConnection(url="http://127.0.0.1:1", api_key="k", hello={}, on_guess=lambda m: None)
    now = time.monotonic()
    # a dropped connection or a timeout: 1, 2, 4, 8, 16 s, then every 30 s
    assert [b._wait_after(a, None, None, False) for a in range(8)] == [1, 2, 4, 8, 16, 30, 30, 30]
    for status in (400, 401, 402, 404, 409):
        assert b._wait_after(0, None, {"status": status}, True) is None, status
    for status in (408, 429, 503, 529):
        assert b._wait_after(2, None, {"status": status}, True) == 4, status
    # retry_after_s wins when it is longer, up to 300 s
    assert b._wait_after(0, None, {"status": 503, "retry_after_s": 45}, True) == 45
    assert b._wait_after(0, None, {"status": 429, "retry_after_s": 1000}, True) == 300
    assert b._wait_after(3, None, {"status": 529, "retry_after_s": 2}, True) == 8
    # a clean close after 60 s or more reopens at once; an earlier or unclean one waits
    assert b._wait_after(0, now - 61, None, True) == 0
    assert b._wait_after(0, now - 10, None, True) == 1
    assert b._wait_after(0, now - 61, None, False) == 1


async def _reopen_run(api, script=TWO_TURNS):
    runner, url = await serve(api)
    try:
        async with aiohttp.ClientSession() as http:
            w = rabel.STT(ScriptedSTT(script), api_key="test", base_url=url, http_session=http)
            await collect(w, wait_ready=w)
    finally:
        await runner.cleanup()
    return [t - api.connected_at[0] for t in api.connected_at]


async def test_reopens_after_503_with_backoff():
    api = FakeAPI(plan=["refuse", "ok"], status=503, error="waking", retry_after=0, v=2)
    at = await _reopen_run(api)
    assert len(at) == 2 and 0.9 <= at[1] <= 1.5, at
    # the second conversation was ready before the second turn began: that turn was sent
    assert ("end_of_turn", "And my savings.") in [(m["type"], m.get("text")) for m in api.received]


async def test_no_reopen_after_400():
    api = FakeAPI(plan=["refuse"], status=400, error="bad_message", retry_after=0, v=2)
    at = await _reopen_run(api)
    assert len(at) == 1, at


async def test_not_ready_within_the_open_timeout_retries(monkeypatch, caplog):
    import livekit.plugins.rabel._bell as bell

    monkeypatch.setattr(bell, "_OPEN_TIMEOUT_S", 0.3)
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    api = FakeAPI(v=2, ready_after=5.0)
    at = await _reopen_run(api)
    # 0.3 s waiting for ready, then the first back-off wait of 1 s
    assert len(at) == 2 and 1.2 <= at[1] <= 1.7, at
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == 1 and "not ready within" in lines[0], lines


@pytest.mark.parametrize("long_s, reopen_at", [(0.2, (0.4, 0.7)), (10.0, (1.35, 1.9))])
async def test_clean_close_reopens_at_once_only_after_a_long_conversation(
    long_s, reopen_at, monkeypatch
):
    import livekit.plugins.rabel._bell as bell

    monkeypatch.setattr(bell, "_LONG_CONVERSATION_S", long_s)
    api = FakeAPI(v=2, close_after=0.4)
    at = await _reopen_run(api)
    assert len(at) >= 2 and reopen_at[0] <= at[1] <= reopen_at[1], at


async def test_contradicting_interim_passes_and_eager_replaces():
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(
            0.30, T.INTERIM_TRANSCRIPT, "check my savings", 0.95
        ),  # rule 4: contradicts "check my balance"
        ev(0.55, T.PREFLIGHT_TRANSCRIPT, "Check my savings.", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Check my savings."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers={"check my": "check my balance"}, v=2))
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "check my"),
        (T.PREFLIGHT_TRANSCRIPT, "check my balance."),
        (T.INTERIM_TRANSCRIPT, "check my savings"),
        (T.PREFLIGHT_TRANSCRIPT, "Check my savings."),
        (T.FINAL_TRANSCRIPT, "Check my savings."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_cancelled_by_words"] == 1


# rule 7: a final with exactly the text of a guess the words cancelled is passed first as an early start on its own
# words (LiveKit remembers the guess's text until the next final and starts no early reply at a final equal to it)
STALE_GUESS = {"check my": "check my balance"}


@pytest.mark.parametrize(
    "middle, final, re_sent",
    [
        pytest.param(
            [("check my bills", 0.95)], "check my balance.", True, id="rule4-cancel-final-equal"
        ),
        pytest.param(
            [("check my bill", 0.40), ("check my bills", 0.95)],
            "check my balance.",
            True,
            id="rule3-given-up-final-equal",
        ),
        pytest.param(
            [("check my bills", 0.95)],
            "Check my balance.",
            False,
            id="final-differs-livekit-starts-its-own",
        ),
        pytest.param([], "check my balance.", False, id="guess-still-live-its-reply-waits"),
    ],
)
async def test_final_equal_to_a_cancelled_guess_goes_first_as_an_early_start(
    middle, final, re_sent
):
    script = [ev(0.00, T.START_OF_SPEECH), ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95)]
    at = 0.30
    for text, conf in middle:
        script.append(ev(at, T.INTERIM_TRANSCRIPT, text, conf))
        at += 0.20
    script += [
        ev(at, T.INTERIM_TRANSCRIPT, "check my balance", 0.95),
        ev(at + 0.20, T.FINAL_TRANSCRIPT, final),
        ev(at + 0.20, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=STALE_GUESS, v=2))
    assert out[:3] == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "check my"),
        (T.PREFLIGHT_TRANSCRIPT, "check my balance."),
    ]
    tail = [(T.PREFLIGHT_TRANSCRIPT, final)] if re_sent else []
    assert out[-(len(tail) + 2) :] == [*tail, (T.FINAL_TRANSCRIPT, final), (T.END_OF_SPEECH, None)]
    assert sum(1 for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT) == 1 + len(tail)
    assert counts.get("final_sent_as_early_start", 0) == len(tail)
    assert counts["guess_sent"] == 1


# rule 8: while LiveKit's own early reply made on a final waits for LiveKit's commit, Bell's guess is held
BEFORE = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(
        0.30, T.FINAL_TRANSCRIPT, "I'd like to check my balance."
    ),  # LiveKit starts its own early reply on it
    ev(0.30, T.END_OF_SPEECH),
]
AFTER = [
    ev(0.60, T.START_OF_SPEECH),
    ev(0.65, T.INTERIM_TRANSCRIPT, "and my", 0.95),
    ev(0.95, T.INTERIM_TRANSCRIPT, "and my savings", 0.95),
    ev(1.20, T.FINAL_TRANSCRIPT, "And my savings please."),
    ev(1.20, T.END_OF_SPEECH),
]
SAVINGS = {"and my": "and my savings please"}


@pytest.mark.parametrize(
    "commit_at, sent, count",
    [
        pytest.param(
            None, False, "held_guess_not_sent_turn_over", id="livekit-reply-waits-all-turn"
        ),
        pytest.param(
            0.80, False, "held_guess_not_sent_commit", id="livekit-commits-after-the-guess"
        ),
        pytest.param(0.45, True, None, id="livekit-committed-before-the-guess"),
    ],
)
async def test_guess_held_while_livekits_own_reply_waits(commit_at, sent, count):
    script = [*BEFORE, *AFTER]
    if commit_at is not None:
        script = sorted(
            [*script, ev(commit_at, COMMIT, "I'd like to check my balance.")], key=lambda e: e[0]
        )
    out, counts = await wrapped_run(script, FakeAPI(answers=SAVINGS, v=2))
    guesses = [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT]
    if sent:
        assert guesses == [(T.PREFLIGHT_TRANSCRIPT, "and my savings please.")]
        assert counts["guess_sent"] == 1 and "guess_held_while_livekit_waits" not in counts
    else:
        assert guesses == []
        assert counts["guess_held_while_livekit_waits"] == 1 and counts[count] == 1
        assert "guess_sent" not in counts
        # no guess is live, so the speech-to-text's own events all pass, in order
        assert out == [(typ, text) for _, typ, text, _ in [*BEFORE, *AFTER]]


async def test_held_guess_refused_when_the_stts_own_early_start_replaces_livekits_reply():
    # rule 8 takes the held guess again at the speech-to-text's own early start; that start came first in the turn, so
    # rule 6 refuses the guess (sent, it would cancel that start), and the events are the speech-to-text's
    script = [
        *BEFORE,
        ev(0.60, T.START_OF_SPEECH),
        ev(0.65, T.INTERIM_TRANSCRIPT, "and my", 0.95),
        ev(
            0.85, T.PREFLIGHT_TRANSCRIPT, "And my", 0.95
        ),  # replaces LiveKit's waiting reply, as without Bell
        ev(1.10, T.INTERIM_TRANSCRIPT, "and my savings", 0.95),
        ev(1.30, T.FINAL_TRANSCRIPT, "And my savings please."),
        ev(1.30, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=SAVINGS, v=2))
    assert out == await plain_run(script)
    assert counts["guess_held_while_livekit_waits"] == 1 and counts["guess_after_stt_start"] == 1
    assert "guess_sent" not in counts


async def test_no_hold_after_a_final_livekit_found_unchanged_from_a_version_2_api():
    # as 1.8.13 (today's production): the speech-to-text's own early start had the final's words, LiveKit makes no early
    # reply of its own at that final, rule 8 holds nothing and the next turn's guess is sent
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.20, T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.", 0.95),
        ev(0.30, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
        ev(0.30, T.END_OF_SPEECH),
        *AFTER,
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=SAVINGS, v=2))
    assert (T.PREFLIGHT_TRANSCRIPT, "and my savings please.") in out
    assert counts["guess_sent"] == 1 and "guess_held_while_livekit_waits" not in counts


async def test_hold_after_a_final_equal_to_the_stts_own_early_start():
    # version 3: the speech-to-text's own early start had the final's words: LiveKit makes no early reply of its own at
    # that final, and the one waiting for its commit is the speech-to-text's; the next turn's guess would replace it, so
    # rule 8 holds it (before 1.9.0, nothing was held here)
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.20, T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.", 0.95),
        ev(0.30, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
        ev(0.30, T.END_OF_SPEECH),
        *AFTER,
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=SAVINGS, v=3))
    assert out == await plain_run(script)
    assert counts["guess_held_while_livekit_waits"] == 1
    assert counts["held_guess_not_sent_turn_over"] == 1 and "guess_sent" not in counts


# Bell's guess kept at a final waits for LiveKit's commit like LiveKit's own early reply; a guess on the
# next turn's words, sent before that commit, replaced it, and when LiveKit then committed the first turn's words alone
# its reply started at the commit (seen in a replayed call, with a guess that was always right)
KEPT_TURN = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.30, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ev(0.30, T.END_OF_SPEECH),
]
KEPT_ANSWERS = {"I'd like to check": "I'd like to check my balance.", **SAVINGS}


@pytest.mark.parametrize(
    "commit_at, sent, count",
    [
        pytest.param(None, False, "held_guess_not_sent_turn_over", id="kept-guess-waits-all-turn"),
        pytest.param(0.80, False, "held_guess_not_sent_commit", id="livekit-commits-after-the-guess"),
        pytest.param(0.45, True, None, id="livekit-committed-before-the-guess"),
    ],
)
async def test_guess_held_while_bells_kept_guess_waits(commit_at, sent, count):
    script = [*KEPT_TURN, *AFTER]
    if commit_at is not None:
        script = sorted(
            [*script, ev(commit_at, COMMIT, "I'd like to check my balance.")], key=lambda e: e[0]
        )
    out, counts = await wrapped_run(script, FakeAPI(answers=KEPT_ANSWERS, v=3))
    guesses = [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT]
    # the first turn's guess is sent and kept: the final equals it
    assert guesses[0] == (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.")
    if sent:
        assert guesses[1:] == [(T.PREFLIGHT_TRANSCRIPT, "and my savings please.")]
        assert counts["guess_sent"] == 2 and "guess_held_while_livekit_waits" not in counts
    else:
        assert guesses[1:] == []
        assert counts["guess_sent"] == 1
        assert counts["guess_held_while_livekit_waits"] == 1 and counts[count] == 1


async def test_a_kept_guess_holds_nothing_from_a_version_2_api():
    # as 1.8.13 (today's production): the next turn's guess is sent though the first turn's kept guess waits
    out, counts = await wrapped_run([*KEPT_TURN, *AFTER], FakeAPI(answers=KEPT_ANSWERS, v=2))
    guesses = [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT]
    assert guesses == [
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance."),
        (T.PREFLIGHT_TRANSCRIPT, "and my savings please."),
    ]
    assert counts["guess_sent"] == 2 and "guess_held_while_livekit_waits" not in counts


@pytest.mark.parametrize(
    "v, hold_line, conf, held",
    [
        pytest.param(1, None, 0.40, True, id="api-v1-default-line-0.9"),
        pytest.param(2, 0.9, 0.80, True, id="api-v2-line-0.9-word-0.80-held"),
        pytest.param(2, 0.3, 0.40, False, id="api-v2-line-0.3-word-0.40-passed"),
        pytest.param(2, "high", 0.80, True, id="api-v2-malformed-line-default-0.9"),
    ],
)
async def test_unsure_last_word_uses_ready_hold_line(v, hold_line, conf, held):
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        # rule 3: only the last word differs, and it is not a beginning of the guess's word (a part-word is rule 2's)
        ev(0.30, T.INTERIM_TRANSCRIPT, "check my bill", conf),
        ev(0.55, T.INTERIM_TRANSCRIPT, "check my balance", 0.95),  # agrees with the guess
        ev(0.70, T.FINAL_TRANSCRIPT, "Check my balance."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"check my": "check my balance"}, v=v, hold_line=hold_line)
    out, counts = await wrapped_run(script, api)
    if held:
        # held one update, then dropped because the next interim agrees: the guess stays the running early start
        assert out == [
            (T.START_OF_SPEECH, None),
            (T.INTERIM_TRANSCRIPT, "check my"),
            (T.PREFLIGHT_TRANSCRIPT, "check my balance."),
            (T.INTERIM_TRANSCRIPT, "check my balance"),
            (T.FINAL_TRANSCRIPT, "Check my balance."),
            (T.END_OF_SPEECH, None),
        ]
        assert counts["tail_held"] == 1 and counts["hold_agreed"] == 1
    else:
        # sure enough under the API's line: rule 4 passes it, which cancels the guess
        assert (T.INTERIM_TRANSCRIPT, "check my bill") in out
        assert "tail_held" not in counts and counts["guess_cancelled_by_words"] == 1


# rule 3 is the API's line and one update; fields `ready` may carry that this client does not take change nothing
HOLD_GUESS = {"check my": "check my balance"}


def _hold_script(*tails):
    """the guess "check my balance" live after "check my"; then interims "check my <tail>" (word, confidence) in turn,
    then "check my balance" (agrees with the guess) and the final"""
    script = [ev(0.00, T.START_OF_SPEECH), ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95)]
    at = 0.30
    for word, conf in tails:
        script.append(ev(at, T.INTERIM_TRANSCRIPT, f"check my {word}", conf))
        at += 0.15
    script += [
        ev(at, T.INTERIM_TRANSCRIPT, "check my balance", 0.95),
        ev(at + 0.15, T.FINAL_TRANSCRIPT, "Check my balance."),
        ev(at + 0.15, T.END_OF_SPEECH),
    ]
    return script


KEPT = [
    (T.START_OF_SPEECH, None),
    (T.INTERIM_TRANSCRIPT, "check my"),
    (T.PREFLIGHT_TRANSCRIPT, "check my balance."),
    (T.INTERIM_TRANSCRIPT, "check my balance"),
    (T.FINAL_TRANSCRIPT, "Check my balance."),
    (T.END_OF_SPEECH, None),
]


def _rabel_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]


@pytest.mark.parametrize(
    "v, extra",
    [
        pytest.param(1, {}, id="api-v1"),
        pytest.param(2, {}, id="api-v2"),
        pytest.param(2, {"hold_always": True, "hold_updates": 2}, id="api-v2-other-fields-ignored"),
        pytest.param(3, {}, id="api-v3-without-the-hold-fields"),
    ],
)
async def test_rule_3_holds_one_unsure_update_at_the_line(v, extra, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    # one unsure last word under the line, then the words agree again: held one update and dropped
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.4)), FakeAPI(answers=HOLD_GUESS, v=v, ready_extra=extra)
    )
    assert out == KEPT and counts["tail_held"] == 1 and counts["hold_agreed"] == 1
    # two unsure updates in a row: the second is passed and cancels the guess
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.4), ("bell", 0.4)),
        FakeAPI(answers=HOLD_GUESS, v=v, ready_extra=extra),
    )
    assert (T.INTERIM_TRANSCRIPT, "check my bell") in out
    assert (T.INTERIM_TRANSCRIPT, "check my bill") not in out
    assert counts["hold_passed"] == 1
    # a last word the speech-to-text is sure of (0.97, over the line 0.9) is not held
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.97)), FakeAPI(answers=HOLD_GUESS, v=v, ready_extra=extra)
    )
    assert "tail_held" not in counts and counts["guess_cancelled_by_words"] == 1
    assert _rabel_lines(caplog) == []


# the hold rule moved (1.9.0): rule 3's two settings from a version 3
# `ready`, `hold_always` (any last word, whatever its confidence) and `hold_updates` (1 or 2). Absent or bad, the
# published rule: the line, one update; a bad value logs one line.


@pytest.mark.parametrize(
    "extra, held, logged",
    [
        pytest.param({}, False, 0, id="absent-the-line-applies"),
        pytest.param({"hold_always": False}, False, 0, id="false-the-line-applies"),
        pytest.param({"hold_always": True}, True, 0, id="true-any-last-word"),
        pytest.param({"hold_always": None}, False, 0, id="null-as-absent"),
        pytest.param({"hold_always": "yes"}, False, 1, id="bad-default-and-one-line"),
        pytest.param({"hold_always": 1}, False, 1, id="a-number-is-bad"),
    ],
)
async def test_hold_always_from_ready_holds_a_sure_last_word(extra, held, logged, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    # a last word the speech-to-text is sure of (0.97, over the line 0.9): today's rule 3 does not hold it
    api = FakeAPI(answers=HOLD_GUESS, v=3, ready_extra=extra)
    out, counts = await wrapped_run(_hold_script(("bill", 0.97)), api)
    if held:
        assert out == KEPT
        assert counts["tail_held"] == 1 and counts["hold_agreed"] == 1
    else:
        assert (T.INTERIM_TRANSCRIPT, "check my bill") in out
        assert "tail_held" not in counts and counts["guess_cancelled_by_words"] == 1
    lines = _rabel_lines(caplog)
    assert len(lines) == logged, lines
    if logged:
        assert "hold_always" in lines[0]


@pytest.mark.parametrize(
    "extra, kept, logged",
    [
        pytest.param({}, False, 0, id="absent-one-update"),
        pytest.param({"hold_updates": 1}, False, 0, id="one-update"),
        pytest.param({"hold_updates": 2}, True, 0, id="two-updates"),
        pytest.param({"hold_updates": 2.0}, True, 0, id="two-as-a-float"),
        pytest.param({"hold_updates": 3}, False, 1, id="three-out-of-range-default-and-one-line"),
        pytest.param({"hold_updates": 0}, False, 1, id="zero-out-of-range"),
        pytest.param({"hold_updates": True}, False, 1, id="true-is-bad"),
        pytest.param({"hold_updates": "2"}, False, 1, id="a-string-is-bad"),
    ],
)
async def test_hold_updates_from_ready(extra, kept, logged, caplog):
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    # two unsure last words in a row ("bill", then "bell"), each under the line, then the words agree again
    api = FakeAPI(answers=HOLD_GUESS, v=3, ready_extra=extra)
    out, counts = await wrapped_run(_hold_script(("bill", 0.4), ("bell", 0.4)), api)
    assert counts["tail_held"] == 1
    if kept:
        # held a second update, then dropped because the next interim agrees: the guess stays the running early start
        assert out == KEPT
        assert counts["tail_held_again"] == 1 and counts["hold_agreed"] == 1
    else:
        # one update only: the second interim is passed and cancels the guess
        assert (T.INTERIM_TRANSCRIPT, "check my bell") in out
        assert (T.INTERIM_TRANSCRIPT, "check my bill") not in out
        assert counts["hold_passed"] == 1 and "tail_held_again" not in counts
    lines = _rabel_lines(caplog)
    assert len(lines) == logged, lines
    if logged:
        assert "hold_updates" in lines[0]


async def test_hold_updates_two_never_holds_a_third_nor_a_sure_word():
    # a third unsure update in a row is passed (it cancels the guess)
    two = {"hold_updates": 2}
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.4), ("bell", 0.4), ("ball", 0.4)),
        FakeAPI(answers=HOLD_GUESS, v=3, ready_extra=two),
    )
    assert (T.INTERIM_TRANSCRIPT, "check my ball") in out and counts["tail_held_again"] == 1
    assert counts["hold_passed"] == 1
    # the second update is held only if it too is unsure (under the line), unless hold_always is true
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.4), ("bell", 0.97)), FakeAPI(answers=HOLD_GUESS, v=3, ready_extra=two)
    )
    assert (T.INTERIM_TRANSCRIPT, "check my bell") in out and "tail_held_again" not in counts
    both = {"hold_updates": 2, "hold_always": True}
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.97), ("bell", 0.97)), FakeAPI(answers=HOLD_GUESS, v=3, ready_extra=both)
    )
    assert out == KEPT and counts["tail_held"] == 1 and counts["tail_held_again"] == 1


async def test_hold_settings_as_the_api_serves_them():
    # the API's own values for a version 3 hello: hold_always true, hold_updates 1. A last word the speech-to-text is
    # sure of is held one update; a second differing update is passed
    served = {"hold_always": True, "hold_updates": 1}
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.97)), FakeAPI(answers=HOLD_GUESS, v=3, ready_extra=served)
    )
    assert out == KEPT and counts["tail_held"] == 1 and counts["hold_agreed"] == 1
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.97), ("bell", 0.97)),
        FakeAPI(answers=HOLD_GUESS, v=3, ready_extra=served),
    )
    assert (T.INTERIM_TRANSCRIPT, "check my bell") in out and counts["hold_passed"] == 1


async def test_hold_settings_absent_from_a_version_1_api_keep_today_rule():
    # a version 1 API sends a bare ready: the line 0.9 and one update, exactly as before the settings existed
    out, counts = await wrapped_run(
        _hold_script(("bill", 0.4), ("bell", 0.4)), FakeAPI(answers=HOLD_GUESS, v=1)
    )
    assert (T.INTERIM_TRANSCRIPT, "check my bell") in out and counts["hold_passed"] == 1
    out, counts = await wrapped_run(_hold_script(("bill", 0.97)), FakeAPI(answers=HOLD_GUESS, v=1))
    assert "tail_held" not in counts and counts["guess_cancelled_by_words"] == 1


async def test_part_word_interims_agree_with_the_guess():
    # some speech-to-text services show a word while it is still being said ("acco" before "account"): an interim whose last word is a beginning
    # of the guess's word in that place agrees with the guess (rule 2), whatever its confidence, and is held back
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "check my acco", 0.95),
        ev(0.40, T.INTERIM_TRANSCRIPT, "check my account", 0.95),
        ev(0.50, T.INTERIM_TRANSCRIPT, "check my account bal", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Check my account balance."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(
        script, FakeAPI(answers={"check my": "check my account balance"}, v=2)
    )
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "check my"),
        (T.PREFLIGHT_TRANSCRIPT, "check my account balance."),
        (T.FINAL_TRANSCRIPT, "Check my account balance."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["part_word_held_back"] == 2 and counts["interim_held_back"] == 1
    assert "guess_cancelled_by_words" not in counts


@pytest.mark.parametrize(
    "said",
    [
        pytest.param("check my sav", id="part-word-of-another-word"),
        pytest.param("check my accountant", id="longer-than-the-guess-word"),
        pytest.param("check our acco", id="earlier-word-differs"),
    ],
)
async def test_part_word_that_disagrees_passes(said):
    # only a beginning of the guess's own word in that place agrees; anything else is rule 4 and cancels the guess
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, said, 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Check my savings."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(
        script, FakeAPI(answers={"check my": "check my account balance"}, v=2)
    )
    assert (T.INTERIM_TRANSCRIPT, said) in out
    assert counts["guess_cancelled_by_words"] == 1 and "part_word_held_back" not in counts


async def test_no_guess_as_third_early_start_and_none_equal_to_words():
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.PREFLIGHT_TRANSCRIPT, "yes", 0.95),
        ev(0.20, T.PREFLIGHT_TRANSCRIPT, "yes please", 0.95),
        ev(0.35, T.INTERIM_TRANSCRIPT, "yes please do", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Yes, please do it."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    for v in (2, 3):
        api = FakeAPI(answers={"yes please do": "yes please do it", "yes": "yes"}, v=v)
        out, counts = await wrapped_run(script, api)
        assert (T.PREFLIGHT_TRANSCRIPT, "yes please do it") not in out
        assert counts["guess_third_start"] == 1
        if v == 2:
            # as 1.8.13: a guess must add words to the words it was made from
            assert counts["guess_equals_words"] == 1
        else:
            # version 3: a guess equal to its words is the end bet, taken; this one equals the speech-to-text's running
            # start
            assert counts["guess_same_as_running"] == 1
        assert out == await plain_run(script)


async def test_guess_reached_by_the_words_before_it_arrives_is_sent():
    # some speech-to-text services show a turn's last words in one burst: the words reach Bell's guess before its answer
    # arrives. The guess (made from fewer words) is then Bell's guess that the turn ends on these words, and is sent as
    # the early start on them; LiveKit keeps it when the final words are the same.
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "my debit", 0.95),
        ev(0.08, T.INTERIM_TRANSCRIPT, "my debit card.", 0.95),
        ev(0.40, T.FINAL_TRANSCRIPT, "my debit card."),
        ev(0.40, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"my debit": "my debit card"}, v=2, delay=0.1)
    out, counts = await wrapped_run(script, api)
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "my debit"),
        (T.INTERIM_TRANSCRIPT, "my debit card."),
        (T.PREFLIGHT_TRANSCRIPT, "my debit card."),
        (T.FINAL_TRANSCRIPT, "my debit card."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_reached_by_words"] == 1 and counts["guess_sent"] == 1
    assert "guess_equals_words" not in counts



# ------------------------------------------------------------------ guessing again (version 3, rule 6a)

# Bell's first guess "...check my balance" is cancelled by the words ("...check my savings"); Bell guesses again from
# those words and the new guess is the next early start
AGAIN = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check my savings", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my savings account", 0.95),
    ev(0.80, T.FINAL_TRANSCRIPT, "I'd like to check my savings account."),
    ev(0.80, T.END_OF_SPEECH),
]
AGAIN_ANSWERS = {
    "I'd like to check": "i'd like to check my balance",
    "I'd like to check my savings": "i'd like to check my savings account",
}


async def test_a_new_guess_after_the_words_cancelled_the_first_is_sent_and_kept():
    api = FakeAPI(answers=AGAIN_ANSWERS, v=3)
    out, counts = await wrapped_run(AGAIN, api)
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like to check"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance."),
        (T.INTERIM_TRANSCRIPT, "I'd like to check my savings"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my savings account."),
        (T.INTERIM_TRANSCRIPT, "I'd like to check my savings account"),
        (T.FINAL_TRANSCRIPT, "I'd like to check my savings account."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_sent"] == 2 and counts["guess_cancelled_by_words"] == 1
    assert "guess_not_first" not in counts
    last = dict(api.received[-1])
    assert isinstance(last.pop("head_start_ms"), int)
    assert last["type"] == "end_of_turn" and last["guess"] == "kept"


async def test_one_guess_alive_a_guess_while_the_first_runs_is_not_sent():
    # the second guess comes while the first is the running early start and the words still agree with it
    api = FakeAPI(answers={**GUESS_AT_CHECK, "I'd like to check my": "i'd like to check my card"}, v=3)
    out, counts = await wrapped_run(TURN, api)
    assert [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT] == [
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.")
    ]
    assert counts["guess_sent"] == 1 and counts["guess_while_live"] == 1


async def test_guessing_again_never_takes_the_third_early_start():
    # two guesses each cancelled by the words have taken two of LiveKit's three early starts: a third is not sent, and
    # the final keeps the last start
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check my savings", 0.95),
        ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my savings balance", 0.95),
        ev(0.80, T.FINAL_TRANSCRIPT, "I'd like to check my savings balance today."),
        ev(0.80, T.END_OF_SPEECH),
    ]
    answers = {
        **AGAIN_ANSWERS,
        "I'd like to check my savings balance": "i'd like to check my savings balance today",
    }
    out, counts = await wrapped_run(script, FakeAPI(answers=answers, v=3))
    assert counts["guess_sent"] == 2 and counts["guess_cancelled_by_words"] == 2
    assert counts["guess_third_start"] == 1
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my savings balance today.") not in out


async def test_no_new_guess_after_the_stts_own_early_start_replaced_bells():
    # the speech-to-text's own early start replaces Bell's running guess: with Bell's guess it has taken two of LiveKit's
    # three early starts, so no new guess is sent (the last start is the final's)
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.20, T.PREFLIGHT_TRANSCRIPT, "I'd like to check my savings", 0.95),
        ev(0.45, T.INTERIM_TRANSCRIPT, "I'd like to check my savings account", 0.95),
        ev(0.80, T.FINAL_TRANSCRIPT, "I'd like to check my savings account."),
        ev(0.80, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=AGAIN_ANSWERS, v=3))
    assert counts["guess_sent"] == 1 and counts["guess_third_start"] == 1
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my savings account.") not in out


@pytest.mark.parametrize(
    "then, sent, waiting_dropped",
    [
        # the hold passes the unsure word: the new guess, made from it, is taken now
        pytest.param("bill", ["check my balance.", "check my bill please."], 0, id="hold-passes-new-guess-taken"),
        # the words agree with the running guess again: it stands and the waiting guess is dropped
        pytest.param("balance", ["check my balance."], 1, id="hold-agrees-waiting-guess-dropped"),
    ],
)
async def test_a_new_guess_during_rule_3s_hold_waits_for_its_outcome(then, sent, waiting_dropped):
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "check my bill", 0.4),
        ev(0.45, T.INTERIM_TRANSCRIPT, f"check my {then}", 0.4),
        ev(0.70, T.FINAL_TRANSCRIPT, "check my bill please." if then == "bill" else "check my balance."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={**HOLD_GUESS, "check my bill": "check my bill please"}, v=3)
    out, counts = await wrapped_run(script, api)
    assert [e[1] for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT] == sent
    assert counts["guess_waits_on_hold"] == 1 and counts["guess_sent"] == len(sent)
    assert counts.get("waiting_guess_dropped", 0) == waiting_dropped
    last = api.received[-1]
    assert last["type"] == "end_of_turn" and last["guess"] == "kept"



# ------------------------------------------------------------------ the end bet (version 3)


async def test_the_end_bet_is_sent_on_the_words_so_far_and_kept():
    # a guess equal to the words it was made from: Bell bets the turn is over there; it is sent as the early start on
    # the words exactly as the speech-to-text showed them, and kept when the turn ends on them
    said = "I'd like to check my balance."
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, said, 0.95),
        ev(0.40, T.FINAL_TRANSCRIPT, said),
        ev(0.40, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={said: "i'd like to check my balance"}, v=3)
    out, counts = await wrapped_run(script, api)
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, said),
        (T.PREFLIGHT_TRANSCRIPT, said),
        (T.FINAL_TRANSCRIPT, said),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_sent"] == 1 and counts["end_bet_sent"] == 1
    assert "guess_equals_words" not in counts and "guess_reached_by_words" not in counts
    last = dict(api.received[-1])
    assert isinstance(last.pop("head_start_ms"), int)
    assert last == {"type": "end_of_turn", "text": said, "guess": "kept"}


async def test_an_end_bet_the_caller_goes_past_is_cancelled_and_bell_may_bet_again():
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check my balance", 0.95),
        ev(0.60, T.FINAL_TRANSCRIPT, "I'd like to check my balance"),
        ev(0.60, T.END_OF_SPEECH),
    ]
    answers = {
        "I'd like to check": "i'd like to check",
        "I'd like to check my balance": "i'd like to check my balance",
    }
    out, counts = await wrapped_run(script, FakeAPI(answers=answers, v=3))
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like to check"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check"),
        (T.INTERIM_TRANSCRIPT, "I'd like to check my balance"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance"),
        (T.FINAL_TRANSCRIPT, "I'd like to check my balance"),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["end_bet_sent"] == 2 and counts["guess_cancelled_by_words"] == 1


# ------------------------------------------------------------------ the ways speech-to-text services send words
#
# Each script gives, with made-up words, the events a LiveKit speech-to-text plugin makes in one of these patterns:
#   Deepgram Flux (livekit-plugins-deepgram, stt_v2.py): StartOfTurn -> START_OF_SPEECH and an interim; each Update -> an
#     interim of the turn so far, word by word, a shown word sometimes changed; EagerEndOfTurn -> PREFLIGHT_TRANSCRIPT;
#     TurnResumed -> an interim of its transcript; EndOfTurn -> FINAL_TRANSCRIPT and END_OF_SPEECH.
#   Part words: each message -> one interim of the words so far, so a word shows while it is still being said ("chan"
#     before "change"); a message whose words are all final -> the plugin's own PREFLIGHT_TRANSCRIPT; the turn's end ->
#     FINAL_TRANSCRIPT and END_OF_SPEECH.
#   Last words in one burst: an interim when the words change; a turn's last words come in one burst, with the service's
#     own early end -> PREFLIGHT_TRANSCRIPT; a resume -> an interim of the same words (it withdraws nothing by itself: the
#     next update moves past them); the turn's end -> FINAL_TRANSCRIPT and END_OF_SPEECH.


@pytest.mark.parametrize("v", [1, 2], ids=["api-v1", "api-v2"])
async def test_one_guess_a_turn_and_no_end_bet_from_an_api_that_is_not_version_3(v):
    # as 1.8.13 (today's production): a second guess in the turn is not sent, though the words cancelled the first
    out, counts = await wrapped_run(AGAIN, FakeAPI(answers=AGAIN_ANSWERS, v=v))
    assert [e[1] for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT] == ["I'd like to check my balance."]
    assert counts["guess_sent"] == 1 and counts["guess_not_first"] == 1
    # a guess equal to the words it was made from is not sent
    said = "I'd like to check my balance."
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, said, 0.95),
        ev(0.40, T.FINAL_TRANSCRIPT, said),
        ev(0.40, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers={said: "i'd like to check my balance"}, v=v))
    assert out == await plain_run(script)
    assert counts["guess_equals_words"] == 1 and "end_bet_sent" not in counts


# ------------------------------------------------------------------ min_chance with the new Bell (version 3)


@pytest.mark.parametrize(
    "stt_kw, chance, sent, dropped",
    [
        # none set: the hello leaves it out, `ready` says null, and the API's balanced setting decides alone
        pytest.param({}, 0.56, True, None, id="none-set-a-guess-under-0.67-is-used"),
        # set: the API raises its bars to it, `ready` says it, and a guess under it is not used
        pytest.param({"min_chance": 0.8}, 0.75, False, "guess_below_min_chance", id="set-0.8-under-it"),
        pytest.param({"min_chance": 0.8}, 0.8, True, None, id="set-0.8-at-it"),
    ],
)
async def test_min_chance_with_a_version_3_api(stt_kw, chance, sent, dropped):
    api = FakeAPI(answers=GUESS_AT_CHECK, v=3, chance=chance)
    out, counts = await wrapped_run(TURN, api, **stt_kw)
    assert ("min_chance" in api.hello) == bool(stt_kw)
    if sent:
        assert counts["guess_sent"] == 1
    else:
        assert out == await plain_run(TURN) and counts[dropped] == 1


async def test_a_version_3_guess_without_chance_is_not_used():
    api = FakeAPI(answers=GUESS_AT_CHECK, v=3, send_chance=False)
    out, counts = await wrapped_run(TURN, api)
    assert out == await plain_run(TURN)
    assert counts["guess_no_chance"] == 1 and "guess_sent" not in counts


async def test_the_api_version_is_read_from_ready():
    for v, want in ((1, False), (2, False), (3, True)):
        api = FakeAPI(v=v)
        runner, url = await serve(api)
        try:
            async with aiohttp.ClientSession() as http:
                w = rabel.STT(ScriptedSTT(TURN), api_key="test", base_url=url, http_session=http)
                stream = w.stream()
                await asyncio.sleep(0.3)
                assert stream._bell is not None and stream._bell.v3 is want
                await stream.aclose()
        finally:
            await runner.cleanup()


async def test_flux_steady_updates_and_its_eager_end_of_turn_on_bells_words():
    # Bell's guess is the running early start. Flux's updates that are a beginning of it are held back (rule 2); an update
    # whose last word Flux changes, under the hold line, is held one update (rule 3) and dropped when the next agrees;
    # Flux's EagerEndOfTurn on the same words is held back (rule 5), so LiveKit keeps the reply already under way.
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "Please", 0.95),
        ev(0.20, T.INTERIM_TRANSCRIPT, "Please send", 0.95),
        ev(0.35, T.INTERIM_TRANSCRIPT, "Please send me", 0.95),
        ev(0.45, T.INTERIM_TRANSCRIPT, "Please send me a few", 0.60),
        ev(0.55, T.INTERIM_TRANSCRIPT, "Please send me a new card", 0.95),
        ev(0.65, T.PREFLIGHT_TRANSCRIPT, "Please send me a new card.", 0.95),
        ev(0.80, T.FINAL_TRANSCRIPT, "Please send me a new card."),
        ev(0.80, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(
        script, FakeAPI(answers={"Please send": "please send me a new card"}, v=2)
    )
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "Please"),
        (T.INTERIM_TRANSCRIPT, "Please send"),
        (T.PREFLIGHT_TRANSCRIPT, "Please send me a new card."),
        (T.INTERIM_TRANSCRIPT, "Please send me a new card"),
        (T.FINAL_TRANSCRIPT, "Please send me a new card."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_sent"] == 1 and counts["interim_held_back"] == 1
    assert counts["tail_held"] == 1 and counts["hold_agreed"] == 1
    assert counts["eager_held_back"] == 1 and "guess_cancelled_by_words" not in counts


async def test_flux_turn_resumed_then_the_caller_goes_on():
    # Flux's EagerEndOfTurn on Bell's words is held back; TurnResumed comes as an interim of the same words and is passed
    # (LiveKit keeps the early reply while the words have not moved past it); the next update moves past the guess and is
    # passed, which cancels it (rule 4); Flux's next EagerEndOfTurn, on other words, is passed as the early start.
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I need", 0.95),
        ev(0.20, T.INTERIM_TRANSCRIPT, "I need to pay", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I need to pay a bill", 0.95),
        ev(0.35, T.PREFLIGHT_TRANSCRIPT, "I need to pay a bill.", 0.95),
        ev(0.45, T.INTERIM_TRANSCRIPT, "I need to pay a bill", 0.95),  # TurnResumed
        ev(0.55, T.INTERIM_TRANSCRIPT, "I need to pay a bill online", 0.95),
        ev(0.65, T.PREFLIGHT_TRANSCRIPT, "I need to pay a bill online.", 0.95),
        ev(0.75, T.FINAL_TRANSCRIPT, "I need to pay a bill online."),
        ev(0.75, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(
        script, FakeAPI(answers={"I need": "i need to pay a bill"}, v=2)
    )
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I need"),
        (T.PREFLIGHT_TRANSCRIPT, "I need to pay a bill."),
        (T.INTERIM_TRANSCRIPT, "I need to pay a bill"),
        (T.INTERIM_TRANSCRIPT, "I need to pay a bill"),
        (T.INTERIM_TRANSCRIPT, "I need to pay a bill online"),
        (T.PREFLIGHT_TRANSCRIPT, "I need to pay a bill online."),
        (T.FINAL_TRANSCRIPT, "I need to pay a bill online."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["eager_held_back"] == 1 and counts["guess_cancelled_by_words"] == 1


async def test_flux_eager_running_then_bells_longer_guess_is_not_a_new_early_start():
    # Flux's eager on (with turn_detection="stt"): its EagerEndOfTurn comes before Bell's answer and is
    # passed, so LiveKit runs an early reply on its words. Bell's guess then arrives with those words and more. Sent, it
    # would be a new early start: LiveKit cancels the reply already running on Flux's words for one on the guess, and when
    # the caller stops where Flux guessed, the final does not match the guess and the reply starts only at LiveKit's
    # commit. So the guess is not passed: LiveKit keeps the running early start, and the events are exactly Flux's.
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I want", 0.95),
        ev(0.20, T.INTERIM_TRANSCRIPT, "I want to pay", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I want to pay my bill", 0.95),
        ev(0.31, T.PREFLIGHT_TRANSCRIPT, "I want to pay my bill.", 0.95),  # EagerEndOfTurn
        ev(0.70, T.FINAL_TRANSCRIPT, "I want to pay my bill."),  # EndOfTurn on the eager's words
        ev(0.70, T.END_OF_SPEECH),
    ]
    # Bell's answer to the words "I want to pay my bill" comes 0.1 s after them, after Flux's EagerEndOfTurn
    api = FakeAPI(answers={"I want to pay my bill": "i want to pay my bill online"}, v=2, delay=0.1)
    out, counts = await wrapped_run(script, api)
    assert [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT] == [
        (T.PREFLIGHT_TRANSCRIPT, "I want to pay my bill.")
    ]
    assert out == await plain_run(script)
    assert "guess_sent" not in counts
    assert counts["guess_after_stt_start"] == 1  # rule 6: Flux's own early start came first in the turn


# Flux's eager end of turn has started a reply and the caller goes on, then says Bell's longer guess or leaves it: the
# guess is not sent either way (rule 6), so the turn is Flux's alone. When the caller says the guess, LiveKit starts on
# Flux's next EagerEndOfTurn: that is the win the rule gives up.
PAY_EAGER = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I want", 0.95),
    ev(0.20, T.INTERIM_TRANSCRIPT, "I want to pay", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I want to pay my bill", 0.95),
    ev(0.31, T.PREFLIGHT_TRANSCRIPT, "I want to pay my bill.", 0.95),  # EagerEndOfTurn
    ev(0.45, T.INTERIM_TRANSCRIPT, "I want to pay my bill", 0.95),  # TurnResumed
]
PAY_GUESS = {"I want to pay my bill": "i want to pay my bill online please"}


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param(
            [
                ev(0.55, T.INTERIM_TRANSCRIPT, "I want to pay my bill online", 0.95),
                ev(0.65, T.INTERIM_TRANSCRIPT, "I want to pay my bill online please", 0.95),
                ev(0.70, T.PREFLIGHT_TRANSCRIPT, "I want to pay my bill online please.", 0.95),
                ev(0.90, T.FINAL_TRANSCRIPT, "I want to pay my bill online please."),
                ev(0.90, T.END_OF_SPEECH),
            ],
            id="the-caller-says-the-guess",
        ),
        pytest.param(
            [
                ev(0.55, T.INTERIM_TRANSCRIPT, "I want to pay my bill today", 0.95),
                ev(0.65, T.PREFLIGHT_TRANSCRIPT, "I want to pay my bill today.", 0.95),
                ev(0.90, T.FINAL_TRANSCRIPT, "I want to pay my bill today."),
                ev(0.90, T.END_OF_SPEECH),
            ],
            id="the-words-leave-the-guess",
        ),
    ],
)
async def test_flux_eager_running_then_bells_longer_guess_whatever_the_caller_says(tail):
    script = [*PAY_EAGER, *tail]
    out, counts = await wrapped_run(script, FakeAPI(answers=PAY_GUESS, v=2, delay=0.1))
    assert out == await plain_run(script)
    assert counts["guess_after_stt_start"] >= 1 and "guess_sent" not in counts


async def test_flux_start_cancelled_then_bells_guess_is_not_the_second_start():
    # A turn seen in measured calls: Flux's first EagerEndOfTurn starts LiveKit's reply and the
    # words move past it; Bell's guess then arrives. Sent, it would be the turn's second early start; Flux's next
    # EagerEndOfTurn, on a beginning of the guess, would replace it as the third, and Flux's last one, on the words the
    # caller ends on, would be refused as over LiveKit's three. So the guess is not sent and Flux keeps its three.
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "Hi my name is John", 0.95),
        ev(0.10, T.PREFLIGHT_TRANSCRIPT, "Hi. My name is John.", 0.95),  # EagerEndOfTurn: start 1
        ev(0.20, T.INTERIM_TRANSCRIPT, "Hi my name is John what are", 0.95),
        ev(0.35, T.INTERIM_TRANSCRIPT, "Hi my name is John what are the branch", 0.95),
        ev(0.40, T.PREFLIGHT_TRANSCRIPT, "Hi. My name is John. What are the branch", 0.95),  # start 2
        ev(0.50, T.INTERIM_TRANSCRIPT, "Hi my name is John what are the branch hours", 0.95),
        ev(0.55, T.PREFLIGHT_TRANSCRIPT, "Hi. My name is John. What are the branch hours", 0.95),  # start 3
        ev(0.60, T.FINAL_TRANSCRIPT, "Hi. My name is John. What are the branch hours"),
        ev(0.60, T.END_OF_SPEECH),
    ]
    api = FakeAPI(
        answers={"Hi my name is John what are": "hi my name is john what are the branch hours"},
        v=2,
        delay=0.1,
    )
    out, counts = await wrapped_run(script, api)
    assert out == await plain_run(script)
    assert counts["guess_after_stt_start"] == 1 and "guess_sent" not in counts


# A turn seen in measured calls: Bell's guess is the turn's first early start; Flux's EagerEndOfTurns replace it and take
# the other two; Flux's last, on the same words as the reply already running (only its punctuation differs), finds no
# start left. Passed, LiveKit would cancel the running reply and start none, and the reply would start at the commit.
NO_START_LEFT = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "The company", 0.95),
    ev(0.15, T.INTERIM_TRANSCRIPT, "The company is", 0.95),
    ev(0.20, T.PREFLIGHT_TRANSCRIPT, "The company is Smart", 0.95),  # replaces Bell's guess: start 2
    ev(0.30, T.INTERIM_TRANSCRIPT, "The company is smart elect", 0.95),
    ev(0.40, T.INTERIM_TRANSCRIPT, "The company is smart elect electric", 0.95),
    ev(0.45, T.PREFLIGHT_TRANSCRIPT, "The company is smart elect electric", 0.95),  # start 3
    ev(0.55, T.PREFLIGHT_TRANSCRIPT, "The company is smart elect electric.", 0.95),  # none left
    ev(0.60, T.FINAL_TRANSCRIPT, "The company is smart elect electric."),
    ev(0.60, T.END_OF_SPEECH),
]


async def test_no_start_left_the_stts_early_start_is_held_back():
    api = FakeAPI(answers={"The company": "the company is smart electric"}, v=2)
    out, counts = await wrapped_run(NO_START_LEFT, api)
    assert counts["guess_sent"] == 1 and counts["eager_held_no_start_left"] == 1
    pre = [t for typ, t in out if typ == T.PREFLIGHT_TRANSCRIPT]
    assert pre[1:] == ["The company is Smart", "The company is smart elect electric"]
    assert out[-2:] == [
        (T.FINAL_TRANSCRIPT, "The company is smart elect electric."),
        (T.END_OF_SPEECH, None),
    ]


async def test_no_start_left_without_bells_start_changes_nothing():
    # with no guess of Bell's in the turn the speech-to-text's early starts all pass, as without Bell
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.10, T.PREFLIGHT_TRANSCRIPT, "The company", 0.95),
        ev(0.20, T.PREFLIGHT_TRANSCRIPT, "The company is Smart", 0.95),
        ev(0.45, T.PREFLIGHT_TRANSCRIPT, "The company is smart elect electric", 0.95),
        ev(0.55, T.PREFLIGHT_TRANSCRIPT, "The company is smart elect electric.", 0.95),
        ev(0.60, T.FINAL_TRANSCRIPT, "The company is smart elect electric."),
        ev(0.60, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers={}, v=2))
    assert out == await plain_run(script)
    assert "eager_held_no_start_left" not in counts


async def test_eager_off_flux_words_as_interims_reach_neither_new_rule():
    # Flux with eager off (and the bench's `--turn livekit` arms, which pass a speech-to-text early start on as an
    # interim): no PREFLIGHT_TRANSCRIPT comes from the speech-to-text, so no early start of its own is counted and
    # neither rule 6's new refusal nor rule 9 is reached; Bell's longer guess is sent as before
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I want", 0.95),
        ev(0.20, T.INTERIM_TRANSCRIPT, "I want to pay", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I want to pay my bill", 0.95),
        ev(0.31, T.INTERIM_TRANSCRIPT, "I want to pay my bill.", 0.95),  # the eager, as an interim
        ev(0.70, T.FINAL_TRANSCRIPT, "I want to pay my bill online."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"I want to pay my bill": "i want to pay my bill online"}, v=2, delay=0.1)
    out, counts = await wrapped_run(script, api)
    pre = [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT]
    assert len(pre) == 1 and words_of(pre[0][1]) == words_of("i want to pay my bill online")
    assert counts["guess_sent"] == 1
    assert "guess_after_stt_start" not in counts and "eager_held_no_start_left" not in counts


async def test_part_words_and_the_services_own_early_start():
    # a service that shows a word while it is said: interims whose last word is a beginning of the guess's word agree with
    # the guess and are held back (rule 2, the part-word); when a message leaves only final words, the plugin makes its own
    # early start, which on the guess's words is held back (rule 5); the turn's end gives the final.
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "Can I", 0.95),
        ev(0.15, T.INTERIM_TRANSCRIPT, "Can I chan", 0.95),
        ev(0.25, T.INTERIM_TRANSCRIPT, "Can I change my", 0.95),
        ev(0.35, T.INTERIM_TRANSCRIPT, "Can I change my add", 0.95),
        ev(0.45, T.PREFLIGHT_TRANSCRIPT, "Can I change my address", 0.95),
        ev(0.55, T.FINAL_TRANSCRIPT, "Can I change my address?"),
        ev(0.55, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(
        script, FakeAPI(answers={"Can I": "can i change my address"}, v=2)
    )
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "Can I"),
        (T.PREFLIGHT_TRANSCRIPT, "Can I change my address?"),
        (T.FINAL_TRANSCRIPT, "Can I change my address?"),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["part_word_held_back"] == 2 and counts["interim_held_back"] == 1
    assert counts["eager_held_back"] == 1 and "guess_cancelled_by_words" not in counts


BURST = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
    # the burst: the service holds the turn's last words and shows them together, faster than Bell's answer
    ev(0.07, T.INTERIM_TRANSCRIPT, "I'd like to close", 0.95),
    ev(0.09, T.INTERIM_TRANSCRIPT, "I'd like to close my account.", 0.95),
]
BURST_GUESS = {"I'd like": "i'd like to close my account"}


async def test_burst_reaches_bells_guess_before_it_arrives():
    # Bell alone (the service's own early start off): the burst shows the guessed words before Bell's answer arrives; the guess is
    # sent as the early start on those words (rule 6, a guess the shown words have caught up with), not dropped.
    script = [
        *BURST,
        ev(0.40, T.FINAL_TRANSCRIPT, "I'd like to close my account."),
        ev(0.40, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=BURST_GUESS, v=2, delay=0.1))
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like"),
        (T.INTERIM_TRANSCRIPT, "I'd like to close"),
        (T.INTERIM_TRANSCRIPT, "I'd like to close my account."),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to close my account."),
        (T.FINAL_TRANSCRIPT, "I'd like to close my account."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_reached_by_words"] == 1 and counts["guess_sent"] == 1


async def test_burst_with_the_services_early_end_then_resume():
    # the service's own early start on: its early end comes with the burst and LiveKit starts on it; Bell's guess, arriving
    # after, has the running early start's words and is not sent (rule 6). The resume's interim and the words after it
    # pass; the hookup changes nothing the service sends.
    script = [
        *BURST,
        ev(0.10, T.PREFLIGHT_TRANSCRIPT, "I'd like to close my account.", 0.95),  # its early end
        ev(0.25, T.INTERIM_TRANSCRIPT, "I'd like to close my account.", 0.95),  # its resume
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to close my account today.", 0.95),
        ev(0.35, T.PREFLIGHT_TRANSCRIPT, "I'd like to close my account today.", 0.95),
        ev(0.45, T.FINAL_TRANSCRIPT, "I'd like to close my account today."),
        ev(0.45, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=BURST_GUESS, v=2, delay=0.1))
    assert out == await plain_run(script)
    assert counts["guess_same_as_running"] == 1 and "guess_sent" not in counts


async def test_services_early_end_on_bells_live_guess_then_resume():
    # Bell's guess is live before the burst: the burst's beginning is held back (rule 2), the service's early end on the
    # guess's words is held back (rule 5); the resume's interim of the same words passes; the update after it moves past the
    # guess and is passed, which cancels it (rule 4); the service's next early end is passed as the early start.
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "Can you", 0.95),
        ev(0.20, T.INTERIM_TRANSCRIPT, "Can you tell me", 0.95),
        ev(0.22, T.INTERIM_TRANSCRIPT, "Can you tell me my balance", 0.95),
        ev(0.23, T.PREFLIGHT_TRANSCRIPT, "Can you tell me my balance", 0.95),  # its early end
        ev(0.30, T.INTERIM_TRANSCRIPT, "Can you tell me my balance", 0.95),  # its resume
        ev(0.40, T.INTERIM_TRANSCRIPT, "Can you tell me my balance please", 0.95),
        ev(0.45, T.PREFLIGHT_TRANSCRIPT, "Can you tell me my balance please", 0.95),
        ev(0.55, T.FINAL_TRANSCRIPT, "Can you tell me my balance please?"),
        ev(0.55, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(
        script, FakeAPI(answers={"Can you": "can you tell me my balance"}, v=2)
    )
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "Can you"),
        (T.PREFLIGHT_TRANSCRIPT, "Can you tell me my balance?"),
        (T.INTERIM_TRANSCRIPT, "Can you tell me my balance"),
        (T.INTERIM_TRANSCRIPT, "Can you tell me my balance"),
        (T.INTERIM_TRANSCRIPT, "Can you tell me my balance please"),
        (T.PREFLIGHT_TRANSCRIPT, "Can you tell me my balance please"),
        (T.FINAL_TRANSCRIPT, "Can you tell me my balance please?"),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["interim_held_back"] == 1 and counts["eager_held_back"] == 1
    assert counts["guess_cancelled_by_words"] == 1


class ContextSTT(ScriptedSTT):
    """A speech-to-text that takes the conversation context, and keeps what it is handed."""

    def __init__(self, script):
        stt.STT.__init__(
            self,
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=True, chat_context=True
            ),
        )
        self.script = script
        self.items: list = []

    def _push_conversation_item(self, ev) -> None:
        self.items.append(ev)


async def test_agent_line_not_sent_and_context_passed_on():
    from livekit.agents import llm
    from livekit.agents.voice.events import ConversationItemAddedEvent

    item = ConversationItemAddedEvent(
        item=llm.ChatMessage(role="assistant", content=["How can I help you today?"])
    )
    # the wrapper always asks for the conversation context (rule 8 sees LiveKit's commit in it); it passes the items on
    # only to a wrapped speech-to-text that takes them
    assert rabel.STT(ScriptedSTT(TURN), api_key="test").capabilities.chat_context
    api = FakeAPI(v=2)
    runner, url = await serve(api)
    try:
        async with aiohttp.ClientSession() as http:
            inner = ContextSTT(TURN)
            w = rabel.STT(inner, api_key="test", base_url=url, http_session=http)
            assert w.capabilities.chat_context
            w._push_conversation_item(item)
            await collect(w, wait_ready=w)
    finally:
        await runner.cleanup()
    assert inner.items == [item]
    heard = [m for m in api.received if m["type"] == "heard"]
    assert heard and all("other" not in m for m in heard)
    assert all(set(m) == {"type", "text"} for m in heard), heard


# ------------------------------------------------------------------ fails open

# (status, error) for each refusal; the README's table lists the same cases
REFUSALS = {
    "400": "bad_message",
    "401": "bad_key",
    "402": "no_seconds",
    "404": "unknown_agent",
    "429": "key_at_once",
    "503": "waking",
    "529": "out_of_capacity",
}


_ANSWERING = [
    ("slow-ready", "api-slow-to-ready"),
    ("slow-guess", "api-slow-guesses"),
    *[(s, f"refused-{s}-{e}") for s, e in REFUSALS.items()],
    ("garbage", "api-errors"),
]


@pytest.mark.parametrize(
    "case, v",
    [
        # no API answers in these, so the API's version does not matter
        pytest.param("down", 2, id="api-down"),
        pytest.param("disabled", 2, id="switched-off"),
        pytest.param("no-key", 2, id="no-key"),
        *[pytest.param(c, v, id=f"{i}-api-v{v}") for c, i in _ANSWERING for v in (1, 2)],
    ],
)
async def test_fails_open(case, v, caplog):
    plain = await plain_run(TURN)
    answers = {
        "I'd like": "i'd like to check my balance",
        "I'd like to check": "i'd like to check my balance",
    }
    runner = None
    url = f"http://127.0.0.1:{free_port()}"
    kw = {}
    if case == "slow-ready":
        api = FakeAPI(answers=answers, ready_after=5.0, v=v)
    elif case == "slow-guess":
        api = FakeAPI(answers=answers, delay=2.0, v=v)  # answers arrive after the turn has ended
    elif case in REFUSALS:
        api = FakeAPI(mode="refuse", status=int(case), error=REFUSALS[case], v=v)
    elif case == "garbage":
        api = FakeAPI(mode="garbage", v=v)
    else:
        api = None
    if case == "disabled":
        kw["enabled"] = False
    if api is not None:
        runner, url = await serve(api)
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    try:
        async with aiohttp.ClientSession() as http:
            w = rabel.STT(
                ScriptedSTT(TURN),
                api_key="" if case == "no-key" else "test",
                base_url=url,
                http_session=http,
                **kw,
            )
            out = await collect(
                w, wait_ready=w, settle={"slow-guess": 2.5, "down": 3.0}.get(case, 0.5)
            )
    finally:
        if runner is not None:
            await runner.cleanup()
    assert out == plain, case
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    if case in ("down", "no-key", *REFUSALS):
        assert len(lines) == 1, lines
    if case == "garbage":  # one line a kind: the unreadable data, and the 400 bad_message
        assert len(lines) == 2, lines
    if case in ("slow-ready", "slow-guess", "disabled"):
        assert lines == []


# ------------------------------------------------------------------ the guess in the final's form (LiveKit's default turn detection)


@pytest.mark.parametrize(
    "shown, added, form",
    [
        pytest.param(
            "I'd like to check", ("my", "balance"), "I'd like to check my balance.", id="statement"
        ),
        pytest.param(
            "Hi. Can you tell me your",
            ("branch", "hours"),
            "Hi. Can you tell me your branch hours?",
            id="question",
        ),
        pytest.param(
            "Thank you.",
            ("im", "all", "set"),
            "Thank you. I'm all set.",
            id="new-sentence-contraction",
        ),
        pytest.param("", ("yes", "i", "dont"), "Yes I don't.", id="no-shown-words"),
        pytest.param("It closes at five", ("pm",), "It closes at five PM.", id="hour-mark"),
        pytest.param("I am", ("ready",), "I am ready.", id="am-as-a-verb"),
    ],
)
def test_final_form(shown, added, form):
    from livekit.plugins.rabel._form import final_form

    text = final_form(shown, added)
    assert text == form
    # the words never change: LiveKit's word comparison still matches the guess
    assert words_of(text) == words_of(" ".join([shown, *added]))


async def test_guess_is_the_final_string_when_its_form_is_right():
    # under LiveKit's default turn detection the early reply is kept only when the final transcript is the same string
    # as the early start (audio_recognition.py, `transcript_changed`); the API's guess is lower case with no punctuation
    api = FakeAPI(answers={"I'd like to check": "id like to check my balance"}, v=2)
    out, counts = await wrapped_run(TURN, api)
    pre = [t for typ, t in out if typ == T.PREFLIGHT_TRANSCRIPT]
    final = [t for typ, t in out if typ == T.FINAL_TRANSCRIPT]
    assert counts["guess_sent"] == 1 and pre == final == ["I'd like to check my balance."]


# ------------------------------------------------------------------ LiveKit's time offset reaches the inner stream


class OffsetSTT(ScriptedSTT):
    """An inner speech-to-text that, like LiveKit's own plugins, adds its stream's start_time_offset to the word times it
    reports: one final whose last word ends 1.0 s after its own first audio, sent after its second audio frame."""

    def __init__(self):
        super().__init__([])

    def stream(self, *, language=NOT_GIVEN, conn_options=DEFAULT_API_CONNECT_OPTIONS):
        return OffsetStream(stt=self, conn_options=conn_options)


class OffsetStream(stt.RecognizeStream):
    async def _run(self) -> None:
        await self._input_ch.recv()
        await self._input_ch.recv()
        off = self.start_time_offset
        word = TimedString("balance", start_time=0.6 + off, end_time=1.0 + off, confidence=0.99)
        alt = stt.SpeechData(
            language=LanguageCode("en"),
            text="balance",
            words=[word],
            confidence=0.9,
            start_time=0.6 + off,
            end_time=1.0 + off,
        )
        self._event_ch.send_nowait(stt.SpeechEvent(type=T.FINAL_TRANSCRIPT, alternatives=[alt]))
        async for _ in self._input_ch:
            pass


@pytest.mark.parametrize("when", ["at-create", "after-first-audio", "never"])
async def test_livekit_time_offset_reaches_the_inner_stream(when):
    # LiveKit's agent sets start_time_offset on the stream it gets from stt.stream() (voice/agent.py), so the vendor's
    # word times count from the session's start; the hookup's stream is not the vendor's, so it must hand the offset on
    w = rabel.STT(OffsetSTT(), enabled=False)
    stream = w.stream()
    if when == "at-create":  # as LiveKit does: right after stream(), before the inner stream exists
        stream.start_time_offset = 2.0
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
    )
    stream.push_frame(frame)
    await asyncio.sleep(0.1)
    if when == "after-first-audio":  # set again while the inner stream already runs
        stream.start_time_offset = 2.0
    stream.push_frame(frame)
    got = []

    async def read():
        async for e in stream:
            got.append(e)

    reader = asyncio.create_task(read())
    await asyncio.sleep(0.2)
    stream.end_input()
    await asyncio.wait_for(reader, 5)
    await stream.aclose()
    finals = [e for e in got if e.type == T.FINAL_TRANSCRIPT]
    assert len(finals) == 1
    end = finals[0].alternatives[0].end_time
    word_end = finals[0].alternatives[0].words[0].end_time
    if when == "never":
        assert stream.start_time_offset == 0.0
        assert end == word_end == pytest.approx(1.0, abs=1e-6)
    else:
        # the inner stream's times are on the hookup's own timeline (the one LiveKit reads), not its first audio's
        assert stream.start_time_offset > 1.5
        assert end == word_end == pytest.approx(1.0 + stream.start_time_offset, abs=0.05)


# ------------------------------------------------------------------ the report on end_of_turn, the two log lines and `user`
# (the API's docs)


async def report_run(script, api, **kw):
    """Like wrapped_run, and also the stream (its summary) and the end_of_turn messages the API received."""
    runner, url = await serve(api)
    try:
        async with aiohttp.ClientSession() as http:
            w = rabel.STT(
                ScriptedSTT(script), api_key="test", base_url=url, http_session=http, **kw
            )
            out = await collect(w, wait_ready=w)
            stream = next(iter(w._streams))
            ends = [m for m in api.received if m["type"] == "end_of_turn"]
            return out, stream, ends
    finally:
        await runner.cleanup()


GUESS_BALANCE = "i'd like to check my balance"

# the words go on past Bell's guess before the final: rule 4 cancels it
DROPPED_BY_WORDS = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
    ev(0.80, T.INTERIM_TRANSCRIPT, "I'd like to check my savings", 0.99),
    ev(0.95, T.FINAL_TRANSCRIPT, "I'd like to check my savings."),
    ev(0.95, T.END_OF_SPEECH),
]
# the words agree with Bell's guess until the final, which adds a word: LiveKit drops the reply started on the guess
DROPPED_AT_FINAL = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
    ev(0.80, T.INTERIM_TRANSCRIPT, "I'd like to check my balance", 0.95),
    ev(0.95, T.FINAL_TRANSCRIPT, "I'd like to check my balance please."),
    ev(0.95, T.END_OF_SPEECH),
]


async def test_report_kept_with_its_head_start():
    api = FakeAPI(answers={"I'd like to check": GUESS_BALANCE}, v=2)
    out, stream, ends = await report_run(TURN, api)
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.") in out
    assert len(ends) == 1
    end = dict(ends[0])
    hs = end.pop("head_start_ms")
    assert end == {"type": "end_of_turn", "text": "I'd like to check my balance.", "guess": "kept"}
    # the guess is passed on about 0.32 s into the turn, the final words arrive at 0.95 s: about 630 ms on the
    # hookup's clock (the script's sleeps and the machine's timer move it by tens of ms)
    assert isinstance(hs, int) and not isinstance(hs, bool) and 450 <= hs <= 900, hs
    assert stream.summary == {
        "turns": 1,
        "guesses": 1,
        "kept": 1,
        "dropped": 0,
        "head_start_ms": hs,
    }


@pytest.mark.parametrize(
    "script",
    [
        pytest.param(DROPPED_BY_WORDS, id="cancelled-by-the-words"),
        pytest.param(DROPPED_AT_FINAL, id="final-words-differ"),
    ],
)
async def test_report_dropped(script):
    api = FakeAPI(answers={"I'd like to check": GUESS_BALANCE}, v=2)
    out, stream, ends = await report_run(script, api)
    assert (
        T.PREFLIGHT_TRANSCRIPT,
        "I'd like to check my balance.",
    ) in out  # the guess was passed on
    assert [(m["guess"], "head_start_ms" in m) for m in ends] == [("dropped", False)]
    assert ends[0]["text"] == script[-2][2]
    assert stream.summary == {
        "turns": 1,
        "guesses": 1,
        "kept": 0,
        "dropped": 1,
        "head_start_ms": 0,
    }


@pytest.mark.parametrize(
    "final, guess",
    [
        # the final leaves LiveKit's remembered early start (the guess's text) unchanged: LiveKit uses the reply
        pytest.param("I'd like to check my balance.", "kept", id="unchanged"),
        # the same words, but LiveKit compares the strings at the final: it starts its own early reply on the final
        # and the one started on Bell's guess is replaced
        pytest.param("I'd like to check my Balance.", "dropped", id="capitals-differ"),
        pytest.param("i'd like to check my balance.", "dropped", id="first-capital-differs"),
        pytest.param("I'd like to check my balance?", "dropped", id="punctuation-differs"),
        pytest.param("I'd like to check my balance", "dropped", id="no-full-stop"),
        pytest.param("I'd like, to check my balance.", "dropped", id="a-comma-added"),
        # on LiveKit's empty turn text both strings are left-stripped first: a leading space changes nothing
        pytest.param(" I'd like to check my balance.", "kept", id="leading-space-first-final"),
    ],
)
async def test_report_follows_livekits_comparison_at_the_final(final, guess):
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
        ev(0.95, T.FINAL_TRANSCRIPT, final),
        ev(0.95, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"I'd like to check": GUESS_BALANCE}, v=2)
    out, stream, ends = await report_run(script, api)
    # Bell's guess was passed on in the final's form, and the final's words equal it in every case
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.") in out
    assert words_of(final) == words_of(GUESS_BALANCE)
    assert len(ends) == 1 and ends[0]["text"] == final
    assert ends[0]["guess"] == guess, ends
    if guess == "kept":
        assert 450 <= ends[0]["head_start_ms"] <= 900, ends
    else:
        assert "head_start_ms" not in ends[0], ends
    assert stream.summary["kept"] == (guess == "kept")
    assert stream.summary["dropped"] == (guess == "dropped")


@pytest.mark.parametrize(
    "api_kw",
    [
        pytest.param({}, id="no-guess"),
        # a guess dropped as slow before it is passed on is none, not dropped
        pytest.param(
            {"answers": {"I'd like to check": GUESS_BALANCE}, "delay": 0.7}, id="slow-guess"
        ),
        # one dropped as wrong data (under min_chance) likewise
        pytest.param(
            {"answers": {"I'd like to check": GUESS_BALANCE}, "chance": 0.5}, id="wrong-data"
        ),
    ],
)
async def test_report_none(api_kw):
    api = FakeAPI(v=2, **api_kw)
    out, stream, ends = await report_run(TURN, api, answer_timeout=0.5)
    assert out == await plain_run(TURN)
    assert ends == [
        {"type": "end_of_turn", "text": "I'd like to check my balance.", "guess": "none"}
    ]
    assert stream.summary == {
        "turns": 1,
        "guesses": 0,
        "kept": 0,
        "dropped": 0,
        "head_start_ms": 0,
    }


@pytest.mark.parametrize("committed", [True, False], ids=["livekit-committed", "not-committed"])
async def test_report_kept_stays_kept_after_livekit_commits(committed):
    # the final equals the guess; if LiveKit commits then, the reply started on the guess is the one it used, and words
    # the caller says after it in the same speech do not undo that; without the commit they cancel it
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
        ev(0.95, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
        *([ev(1.05, COMMIT, "I'd like to check my balance.")] if committed else []),
        ev(1.30, T.INTERIM_TRANSCRIPT, "and also", 0.99),
        ev(1.50, T.FINAL_TRANSCRIPT, "And also."),
        ev(1.50, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"I'd like to check": GUESS_BALANCE}, v=2)
    out, stream, ends = await report_run(script, api)
    assert len(ends) == 1
    if committed:
        assert ends[0]["guess"] == "kept" and 450 <= ends[0]["head_start_ms"] <= 900, ends
    else:
        assert ends[0]["guess"] == "dropped" and "head_start_ms" not in ends[0], ends


@pytest.mark.parametrize(
    "v, script", [(2, TWO_TURNS), (3, TWO_TURNS_COMMITTED)], ids=["api-v2", "api-v3"]
)
async def test_log_lines_kept_and_summary_carry_no_words(v, script, caplog):
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.rabel")
    answers = {"I'd like to check": GUESS_BALANCE, "and my": "and my savings please"}
    api = FakeAPI(answers=answers, v=v)
    out, stream, ends = await report_run(script, api)
    assert [m["guess"] for m in ends] == ["kept", "dropped"]
    hs = ends[0]["head_start_ms"]
    recs = [r for r in caplog.records if r.name == "livekit.plugins.rabel"]
    # one line a kept guess, in the form and at the level of LiveKit's "using preemptive generation" (debug, the
    # number in `extra`)
    kept = [r for r in recs if r.getMessage() == "Bell's guess kept"]
    assert len(kept) == 1 and kept[0].levelno == logging.DEBUG and kept[0].head_start_ms == hs
    # one summary when the stream closes
    summary = [
        r
        for r in recs
        if r.getMessage().startswith("Bell: ") and "head start in all" in r.getMessage()
    ]
    assert len(summary) == 1 and summary[0].levelno == logging.INFO
    s = summary[0]
    assert (s.turns, s.guesses, s.kept, s.dropped, s.head_start_ms_total) == (2, 2, 1, 1, hs)
    assert (
        s.getMessage() == f"Bell: 2 turns, 2 guesses, 1 kept, 1 dropped, {hs} ms head start in all"
    )
    # no word of the call in any line of the hookup's: not the words heard, the finals, nor Bell's guesses
    said = set()
    for _, _, text, _ in TWO_TURNS:
        said |= set(words_of(text))
    for g in answers.values():
        said |= set(words_of(g))
    base = set(vars(logging.makeLogRecord({})))
    for r in recs:
        assert not set(words_of(r.getMessage())) & said, r.getMessage()
        for k, val in vars(r).items():
            if k not in base and isinstance(val, str):
                assert not set(words_of(val)) & said, (k, val)



# the status with more than one guess a turn (version 3): the report's `guess` is the last guess's, `dropped_before` the
# guesses dropped before it; the summary counts every guess
THIRD_REFUSED = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check my savings", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my savings balance", 0.95),
    ev(0.80, T.FINAL_TRANSCRIPT, "I'd like to check my savings balance."),
    ev(0.80, T.END_OF_SPEECH),
]


@pytest.mark.parametrize(
    "script, last, summary",
    [
        # the first dropped by the words, the second kept
        pytest.param(AGAIN, "kept", (1, 2, 1, 1), id="dropped-then-kept"),
        # both dropped by the words
        pytest.param(THIRD_REFUSED, "dropped", (1, 2, 0, 2), id="dropped-twice"),
    ],
)
async def test_report_and_summary_count_every_guess_of_a_turn(script, last, summary, caplog):
    caplog.set_level(logging.INFO, logger="livekit.plugins.rabel")
    api = FakeAPI(answers=AGAIN_ANSWERS, v=3)
    out, stream, ends = await report_run(script, api)
    assert len(ends) == 1
    end = dict(ends[0])
    hs = end.pop("head_start_ms", None)
    assert (hs is not None) == (last == "kept")
    assert end == {"type": "end_of_turn", "text": script[-2][2], "guess": last, "dropped_before": 1}
    s = stream.summary
    assert (s["turns"], s["guesses"], s["kept"], s["dropped"]) == summary
    line = [r for r in caplog.records if "head start in all" in r.getMessage()]
    assert len(line) == 1 and (line[0].guesses, line[0].kept, line[0].dropped) == summary[1:]


async def test_report_with_one_guess_a_turn_is_as_published():
    # one guess: no `dropped_before`, the report exactly as the published hookup sends it
    api = FakeAPI(answers=GUESS_AT_CHECK, v=3)
    out, stream, ends = await report_run(TURN, api)
    assert len(ends) == 1 and "dropped_before" not in ends[0] and ends[0]["guess"] == "kept"
    assert (stream.summary["guesses"], stream.summary["kept"], stream.summary["dropped"]) == (1, 1, 0)


async def test_report_off_leaves_the_field_out():
    api = FakeAPI(answers={"I'd like to check": GUESS_BALANCE}, v=2)
    out, stream, ends = await report_run(TURN, api, report=False)
    assert ends == [{"type": "end_of_turn", "text": "I'd like to check my balance."}]
    # the turn itself is the same with the report on or off
    api_on = FakeAPI(answers={"I'd like to check": GUESS_BALANCE}, v=2)
    out_on, _, _ = await report_run(TURN, api_on)
    assert out == out_on
    w = rabel.STT(ScriptedSTT(TURN), api_key="test")
    assert w._report is True
    w.update_options(report=False)
    assert w._report is False
    for bad in ("no", None, 0):
        with pytest.raises(ValueError):
            rabel.STT(ScriptedSTT(TURN), api_key="test", report=bad)
        with pytest.raises(ValueError):
            w.update_options(report=bad)
    assert w._report is False


async def test_an_api_that_refuses_the_report_fails_open(caplog):
    # no API refuses it (one that does not know the field ignores it); if one did, the stream passes through as it does
    # after any 400: the turn already ended is as it was, and the rest of the call is the speech-to-text's own events
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    answers = {"I'd like to check": GUESS_BALANCE, "and my": "and my savings please"}
    api = FakeAPI(answers=answers, v=2, refuse_report=True)
    out, stream, ends = await report_run(TWO_TURNS, api)
    plain = await plain_run(TWO_TURNS)

    def second_turn(evs):
        starts = [i for i, e in enumerate(evs) if e[0] == T.START_OF_SPEECH]
        return evs[starts[1] :]

    first = out[: len(out) - len(second_turn(out))]
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.") in first
    assert second_turn(out) == second_turn(plain)
    assert len(api.connected_at) == 1  # a 400 opens no other conversation (the API's docs)
    assert [m["guess"] for m in ends] == ["kept"]
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == 1 and "400 bad_message" in lines[0], lines


async def test_user_sent_in_the_hello_when_set():
    api = FakeAPI(v=2)
    runner, url = await serve(api)
    try:
        async with aiohttp.ClientSession() as http:
            w = rabel.STT(
                ScriptedSTT(TURN),
                api_key="test",
                base_url=url,
                http_session=http,
                user="cust-42:a_b.c",
            )
            await collect(w, wait_ready=w)
            assert api.hello == {**hello_v3(), "user": "cust-42:a_b.c"}
            w.update_options(user=None)  # from the next stream, not sent
            await collect(w, wait_ready=w)
            assert api.hello == hello_v3()
            w.update_options(user="Z" * 128)
            await collect(w, wait_ready=w)
            assert api.hello == {**hello_v3(), "user": "Z" * 128}
    finally:
        await runner.cleanup()


@pytest.mark.parametrize(
    "bad",
    ["", "a" * 129, "Jane Smith", "jane@example.com", "+15551234567", "caller/7", 42, True],
    ids=["empty", "129", "a-name", "an-address", "plus-number", "slash", "int", "bool"],
)
async def test_user_form_checked_and_not_repeated(bad):
    with pytest.raises(ValueError) as e:
        rabel.STT(ScriptedSTT(TURN), api_key="test", user=bad)
    if bad != "":
        assert str(bad) not in str(e.value) and repr(bad) not in str(e.value)
    w = rabel.STT(ScriptedSTT(TURN), api_key="test", user="u-1")
    with pytest.raises(ValueError):
        w.update_options(user=bad)
    assert w._hello()["user"] == "u-1"
    assert "user" not in rabel.STT(ScriptedSTT(TURN), api_key="test")._hello()


# ------------------------------------------------------------------ the session's own settings (rule 6)


def in_session(monkeypatch, session):
    """The job's primary session, as `get_job_context` gives it to the wrapper inside a running job."""
    import types

    from livekit.plugins.rabel import stt as rabel_stt

    monkeypatch.setattr(
        rabel_stt,
        "get_job_context",
        lambda *, required=True: types.SimpleNamespace(primary_session=session),
    )


def session_with(**turn_handling):
    from livekit.agents import AgentSession

    return AgentSession(turn_handling=turn_handling)


async def test_the_sessions_settings_are_read_and_the_agents_come_first(monkeypatch):
    import types

    from livekit.agents import Agent
    from livekit.plugins.rabel import stt as rabel_stt
    from livekit.plugins.rabel.stt import _host_limits, _HostLimits

    defaults = _HostLimits(early_replies=True, max_starts=3, max_speech_s=10.0)
    # no job (these tests, a developer's own script): LiveKit's defaults
    assert _host_limits() == defaults
    in_session(monkeypatch, session_with())
    assert _host_limits() == defaults

    in_session(
        monkeypatch,
        session_with(preemptive_generation={"max_retries": 1, "max_speech_duration": 4}),
    )
    assert _host_limits() == _HostLimits(early_replies=True, max_starts=1, max_speech_s=4.0)

    # where LiveKit starts no early reply: preemptive generation off, turns ended by hand, a realtime model
    in_session(monkeypatch, session_with(preemptive_generation={"enabled": False}))
    assert _host_limits().early_replies is False
    in_session(monkeypatch, session_with(turn_detection="manual"))
    assert _host_limits().early_replies is False
    session = session_with()
    not_an_llm = object()  # stands for a realtime model: anything that is not an llm.LLM
    in_session(
        monkeypatch,
        types.SimpleNamespace(
            options=session.options,
            turn_detection=None,
            llm=not_an_llm,
            current_agent=None,
            current_speech=None,
        ),
    )
    assert _host_limits().early_replies is False

    # the running agent's own settings come before the session's, as LiveKit resolves them
    session = session_with(preemptive_generation={"max_retries": 2})
    agent = Agent(
        instructions="x",
        turn_handling={"preemptive_generation": {"max_retries": 1}, "turn_detection": "manual"},
    )
    in_session(
        monkeypatch,
        types.SimpleNamespace(
            options=session.options,
            turn_detection=session.turn_detection,
            llm=None,
            current_agent=agent,
            current_speech=None,
        ),
    )
    assert _host_limits() == _HostLimits(early_replies=False, max_starts=1, max_speech_s=10.0)

    # a job with no session started, or settings that cannot be read: LiveKit's defaults, never an error
    class NoSession:
        @property
        def primary_session(self):
            raise RuntimeError("No AgentSession was started for this job")

    monkeypatch.setattr(rabel_stt, "get_job_context", lambda *, required=True: NoSession())
    assert _host_limits() == defaults
    in_session(monkeypatch, object())
    assert _host_limits() == defaults


@pytest.mark.parametrize(
    "turn_handling",
    [{"preemptive_generation": {"enabled": False}}, {"turn_detection": "manual"}],
    ids=["early-replies-off", "turns-ended-by-hand"],
)
async def test_no_guess_where_livekit_starts_no_early_reply(turn_handling, monkeypatch):
    # a guess would start nothing there; it would only stand in LiveKit's text in place of the caller's words, and a
    # turn ended by hand (`session.commit_user_turn()`) would commit it as what the caller said
    in_session(monkeypatch, session_with(**turn_handling))
    api = FakeAPI(answers=GUESS_AT_CHECK, v=3)
    out, counts = await wrapped_run(TURN, api)
    assert out == await plain_run(TURN)
    assert counts["guess_no_early_reply"] == 1 and "guess_sent" not in counts
    # Bell still hears the turn, and the report says no guess was passed on
    assert [m["type"] for m in api.received].count("heard") == 4
    assert api.received[-1] == {
        "type": "end_of_turn",
        "text": "I'd like to check my balance.",
        "guess": "none",
    }


async def test_with_one_early_start_a_turn_bell_sends_no_guess(monkeypatch):
    # `max_retries` 1: a guess would take the turn's only early start; were it wrong, LiveKit could start none on the
    # final words, and the reply would start later than without Bell
    in_session(monkeypatch, session_with(preemptive_generation={"max_retries": 1}))
    out, counts = await wrapped_run(TURN, FakeAPI(answers=GUESS_AT_CHECK, v=3))
    assert out == await plain_run(TURN)
    assert counts["guess_third_start"] == 1 and "guess_sent" not in counts


async def test_with_two_early_starts_a_turn_bell_takes_only_the_first(monkeypatch):
    in_session(monkeypatch, session_with(preemptive_generation={"max_retries": 2}))
    out, counts = await wrapped_run(TURN, FakeAPI(answers=GUESS_AT_CHECK, v=3))
    assert (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.") in out
    assert counts["guess_sent"] == 1
    # after the speech-to-text's own early start one is left, and it is not Bell's
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.PREFLIGHT_TRANSCRIPT, "I'd like", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
        ev(0.90, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
        ev(0.90, T.END_OF_SPEECH),
    ]
    out, counts = await wrapped_run(script, FakeAPI(answers=GUESS_AT_CHECK, v=3))
    assert out == await plain_run(script)
    assert counts["guess_third_start"] == 1 and "guess_sent" not in counts


async def test_no_guess_after_the_sessions_limit_of_speech(monkeypatch):
    in_session(monkeypatch, session_with(preemptive_generation={"max_speech_duration": 0.2}))
    out, counts = await wrapped_run(TURN, FakeAPI(answers=GUESS_AT_CHECK, v=3))
    assert out == await plain_run(TURN)
    assert counts["guess_past_10s"] == 1 and "guess_sent" not in counts


async def test_a_started_session_is_read_through_its_job():
    # no stand-in for LiveKit here: a real session, started inside a job as LiveKit's own test helper makes one
    from livekit.agents import Agent, AgentSession
    from livekit.agents.testing import fake_job_context
    from livekit.plugins.rabel.stt import _host_limits, _HostLimits

    with fake_job_context():
        session = AgentSession(turn_handling={"preemptive_generation": {"max_retries": 1}})
        await session.start(Agent(instructions="x", turn_handling={"turn_detection": "manual"}))
        try:
            assert _host_limits() == _HostLimits(
                early_replies=False, max_starts=1, max_speech_s=10.0
            )
        finally:
            await session.aclose()
    assert _host_limits() == _HostLimits()


@pytest.mark.parametrize("interrupted, sent", [(False, False), (True, True)], ids=["speaking", "interrupted"])
async def test_no_guess_while_the_agents_reply_is_under_way(interrupted, sent, monkeypatch):
    # LiveKit starts no early reply while the agent's reply is under way and not interrupted; a guess then would start
    # nothing, and its words would count as the caller's where LiveKit counts the words that interrupt the agent. Once
    # the caller has interrupted the reply, a guess is an early start again.
    import types

    session = session_with()
    in_session(
        monkeypatch,
        types.SimpleNamespace(
            options=session.options,
            turn_detection=session.turn_detection,
            llm=None,
            current_agent=None,
            current_speech=types.SimpleNamespace(interrupted=interrupted),
        ),
    )
    out, counts = await wrapped_run(TURN, FakeAPI(answers=GUESS_AT_CHECK, v=3))
    if sent:
        assert counts["guess_sent"] == 1 and "guess_while_agent_speaks" not in counts
    else:
        assert out == await plain_run(TURN)
        assert counts["guess_while_agent_speaks"] == 1 and "guess_sent" not in counts


async def run_in(language, script, api):
    """wrapped_run with a speech-to-text that says its words are in `language`"""
    runner, url = await serve(api)
    try:
        async with aiohttp.ClientSession() as http:
            w = rabel.STT(
                ScriptedSTT(script, language=language),
                api_key="test",
                base_url=url,
                http_session=http,
            )
            out = await collect(w, wait_ready=w)
            ends = [m for m in api.received if m["type"] == "end_of_turn"]
            return out, ends
    finally:
        await runner.cleanup()


# ------------------------------------------------------------------ a language other than English
#
# Under LiveKit's default turn detection an early reply is kept only when the final transcript is the same string as the
# early start's, so the guess is written the way the final will be. For English that is `final_form`'s rules. For any
# other language it is the call's own form: how this call's final transcripts have been written so far.

FIRST_HI = "\u092e\u0947\u0930\u093e \u0928\u093e\u092e \u0930\u093e\u092e \u0939\u0948\u0964"
HEARD_HI = "\u092e\u0941\u091d\u0947 \u0905\u092a\u0928\u093e \u092c\u0948\u0932\u0947\u0902\u0938"
REST_HI = "\u091a\u0947\u0915 \u0915\u0930\u0928\u093e \u0939\u0948"
SAID_HI = HEARD_HI + " " + REST_HI + "\u0964"  # the turn as the speech-to-text writes it, closed with Hindi's own mark
HINDI = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, FIRST_HI[:8], 0.95),
    ev(0.30, T.FINAL_TRANSCRIPT, FIRST_HI),
    ev(0.30, T.END_OF_SPEECH),
    ev(0.45, COMMIT, FIRST_HI),
    ev(0.60, T.START_OF_SPEECH),
    ev(0.65, T.INTERIM_TRANSCRIPT, HEARD_HI, 0.95),
    ev(1.10, T.FINAL_TRANSCRIPT, SAID_HI),
    ev(1.10, T.END_OF_SPEECH),
]


@pytest.mark.parametrize("language, kept", [("hi", True), ("en", False), ("multi", False)])
async def test_another_language_is_written_as_the_calls_own_finals_were(language, kept):
    api = FakeAPI(answers={HEARD_HI: HEARD_HI + " " + REST_HI}, v=3)
    out, ends = await run_in(language, HINDI, api)
    guess = [text for typ, text in out if typ == T.PREFLIGHT_TRANSCRIPT]
    if kept:
        # the call's first final closed with Hindi's own mark, so the guess does, and it is the final's string
        assert guess == [SAID_HI] and ends[-1]["guess"] == "kept"
    else:
        # English, or a speech-to-text that names no one language: the English rules, as before
        assert guess == [HEARD_HI + " " + REST_HI + "."] and ends[-1]["guess"] == "dropped"


def test_the_calls_own_form():
    from livekit.plugins.rabel._form import CallForm, final_form, written

    form = CallForm()
    # before any final transcript there is nothing to copy: the guess is written as `final_form` writes it
    assert written("Je voudrais", ("un", "billet"), form) == final_form("Je voudrais", ("un", "billet"))
    form.final("Je voudrais v\u00e9rifier mon solde.")
    form.final("Pouvez-vous l'emmener \u00e0 l'h\u00f4tel ?")
    # a word as this call wrote it (its apostrophe back), and the turn closed as it closed after the same opening word
    assert written("Pouvez-vous", ("lemmener",), form) == "Pouvez-vous l'emmener ?"
    assert written("Je voudrais", ("un", "billet"), form) == "Je voudrais un billet."
    # an opening word not heard yet: the way most of the call's turns have closed
    form.final("Combien co\u00fbte le billet ?")
    assert written("Quel est", ("le", "prix"), form) == "Quel est le prix ?"
    # a guess that adds nothing (the end bet) is the speech-to-text's own text
    assert written("Je voudrais un billet", (), form) == "Je voudrais un billet"
    # a speech-to-text that writes no capitals and no marks is copied as it is
    plain = CallForm()
    plain.final("mera naam ram hai")
    assert written("", ("mujhe", "balance"), plain) == "mujhe balance"
