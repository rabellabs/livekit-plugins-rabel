"""livekit-plugins-rabel: the wrapper's rules on a scripted speech-to-text against an in-memory stand-in of the Rabel
API (versions 1 and 2 of its conversation), and that it fails open (API down, slow, refusing, erroring: the events
are exactly the inner speech-to-text's).

No network: the stand-in is handed to the plugin as its `http_session`, and its WebSocket is a pair of queues. The
tests run on the virtual-time loop (tests/virtual_time.py). The plugin measures how late a guess is with
`time.monotonic()`, which that loop leaves on the real clock, so `_loop_monotonic` points it at the loop's clock for
these tests. The scripts' words are made up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import LanguageCode, __version__ as agents_version, llm, stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, TimedString
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.plugins import rabel
from livekit.plugins.rabel.stt import words_of

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

T = stt.SpeechEventType


@pytest.fixture(autouse=True)
def _loop_monotonic(monkeypatch: pytest.MonkeyPatch) -> None:
    # resolved at call time, so it reads the virtual clock the test's own fixture installs
    monkeypatch.setattr(time, "monotonic", lambda: time.perf_counter())


# ------------------------------------------------------------------ a scripted inner speech-to-text

# in a script: LiveKit commits a turn, which the wrapper sees as the caller's message added to the conversation
COMMIT = "livekit_commit"


def ev(at: float, typ: Any, text: str | None = None, conf: float | None = None) -> tuple:
    return (at, typ, text, conf)


class ScriptedSTT(stt.STT):
    """Emits (seconds after the first audio frame, event type, text, last word's confidence) in order."""

    def __init__(self, script: list[tuple]) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self.script = script

    @property
    def provider(self) -> str:
        return "Scripted"

    @property
    def model(self) -> str:
        return "Script-1"

    async def _recognize_impl(
        self, buffer: Any, *, language: Any = NOT_GIVEN, conn_options: Any = None
    ) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(
        self, *, language: Any = NOT_GIVEN, conn_options: Any = DEFAULT_API_CONNECT_OPTIONS
    ) -> ScriptedStream:
        return ScriptedStream(stt=self, conn_options=conn_options)


class ScriptedStream(stt.RecognizeStream):
    async def _run(self) -> None:
        await self._input_ch.recv()
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for at, typ, text, conf in self._stt.script:  # type: ignore[attr-defined]
            if typ == COMMIT:
                continue
            wait = t0 + at - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            alts = []
            if text is not None:
                words = [TimedString(w, confidence=0.99) for w in text.split()]
                if words and conf is not None:
                    words[-1] = TimedString(str(words[-1]), confidence=conf)
                alts = [
                    stt.SpeechData(
                        language=LanguageCode("en"), text=text, words=words, confidence=0.9
                    )
                ]
            self._event_ch.send_nowait(stt.SpeechEvent(type=typ, alternatives=alts))
        async for _ in self._input_ch:
            pass


def user_item(text: str) -> ConversationItemAddedEvent:
    return ConversationItemAddedEvent(item=llm.ChatMessage(role="user", content=[text or ""]))


# ------------------------------------------------------------------ an in-memory stand-in of the Rabel API


class FakeWS:
    """The client's end of one conversation: what it sends goes to the stand-in at once, what the stand-in sends
    waits in a queue for `receive()`."""

    def __init__(self, api: FakeAPI) -> None:
        self._api = api
        self._in: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        self.closed = False
        self.close_code: int | None = None
        self.got_hello = False

    async def send_str(self, data: str) -> None:
        if self.closed:
            raise ConnectionResetError("closed")
        self._api.on_message(self, json.loads(data))

    async def receive(self) -> aiohttp.WSMessage:
        return await self._in.get()

    async def close(self) -> bool:
        self._end()
        return True

    def push(self, m: dict | str) -> None:
        if not self.closed:
            data = m if isinstance(m, str) else json.dumps(m)
            self._in.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, data, None))

    def _end(self) -> None:
        if not self.closed:
            self.closed = True
            self.close_code = aiohttp.WSCloseCode.OK
            self._in.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, None))


class FakeSession:
    """Stands in for the `aiohttp.ClientSession` the plugin is given. With no API, connecting fails as a refused
    connection does."""

    closed = False

    def __init__(self, api: FakeAPI | None) -> None:
        self._api = api

    async def ws_connect(self, url: str, **kw: Any) -> FakeWS:
        if self._api is None:
            raise aiohttp.ClientConnectionError("connection refused")
        return self._api.connect()


class FakeAPI:
    """v: 1 answers as the API served before version 2 (a bare `ready`, guesses without `chance`); 2 as version 2
    (`ready` with `min_chance` and `hold_line`, guesses with `chance`). mode: "ok" (guess `answers[heard words]`
    after `delay` s), "refuse" (an error message, then close), "garbage" (broken JSON, a guess for words never sent
    and a 400). `plan` gives a mode for each conversation in turn, the last repeating."""

    def __init__(
        self,
        mode: str = "ok",
        answers: dict[str, str] | None = None,
        delay: float = 0.02,
        status: int = 402,
        error: str = "no_seconds",
        retry_after: float = 60,
        ready_after: float = 0.0,
        v: int = 2,
        chance: float = 0.86,
        send_chance: bool = True,
        min_chance: float | None = None,
        plan: list[str] | None = None,
    ) -> None:
        self.answers, self.delay = answers or {}, delay
        self.status, self.error, self.retry_after = status, error, retry_after
        self.ready_after, self.v = ready_after, v
        self.chance, self.send_chance, self.min_chance = chance, send_chance, min_chance
        self.plan = plan or [mode]
        self.received: list[dict] = []
        self.hello: dict | None = None
        self.connected_at: list[float] = []

    def connect(self) -> FakeWS:
        self.connected_at.append(asyncio.get_running_loop().time())
        return FakeWS(self)

    def _mode(self) -> str:
        return self.plan[min(len(self.connected_at), len(self.plan)) - 1]

    def _ready(self) -> dict:
        if self.v == 1:
            return {"type": "ready"}
        h = self.hello or {}
        min_chance = self.min_chance
        if min_chance is None:  # the API clamps the hello's min_chance and says which is in force
            min_chance = min(max(h.get("min_chance", 0.7), 0.67), 0.99)
        return {"type": "ready", "v": 2, "min_chance": min_chance, "hold_line": 0.9}

    def _guess(self, text: str, heard: str) -> dict:
        g = {"type": "guess", "turn": 0, "id": 1, "text": text, "heard": heard}
        if self.v == 2 and self.send_chance:
            g["chance"] = self.chance
        return g

    def on_message(self, ws: FakeWS, m: dict) -> None:
        loop = asyncio.get_running_loop()
        mode = self._mode()
        if not ws.got_hello:
            ws.got_hello = True
            self.hello = m
            if mode == "refuse":
                ws.push(
                    {
                        "type": "error",
                        "status": self.status,
                        "error": self.error,
                        "message": "refused",
                        "retry_after_s": self.retry_after,
                    }
                )
                ws._end()
            else:
                loop.call_later(self.ready_after, ws.push, self._ready())
            return
        self.received.append(m)
        if mode == "garbage":
            ws.push("{not json")
            ws.push(self._guess("words never heard", "nothing"))
            ws.push({"type": "error", "status": 400, "error": "bad_message", "message": "x"})
            ws._end()
            return
        if m.get("type") == "heard" and m["text"] in self.answers:
            loop.call_later(self.delay, ws.push, self._guess(self.answers[m["text"]], m["text"]))


async def collect(s: stt.STT, settle: float = 0.3) -> list[tuple]:
    stream = s.stream()
    if isinstance(s, rabel.STT):
        await asyncio.sleep(0.2)  # the API's conversation is ready before the first words
    stream.push_frame(
        rtc.AudioFrame(
            data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
        )
    )
    out: list[tuple] = []

    async def read() -> None:
        async for e in stream:
            out.append((e.type, e.alternatives[0].text if e.alternatives else None))

    reader = asyncio.create_task(read())
    script = s.wrapped_stt.script if isinstance(s, rabel.STT) else s.script  # type: ignore[attr-defined]
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


def wrapper(script: list[tuple], api: FakeAPI | None, **kw: Any) -> rabel.STT:
    return rabel.STT(
        ScriptedSTT(script),
        api_key=kw.pop("api_key", "test"),
        base_url="https://api.example",
        http_session=FakeSession(api),  # type: ignore[arg-type]
        **kw,
    )


async def wrapped_run(script: list[tuple], api: FakeAPI | None, settle: float = 0.3, **kw: Any):
    """The events out, the stream's counts and summary, and the end_of_turn messages the API received."""
    w = wrapper(script, api, **kw)
    out = await collect(w, settle=settle)
    stream = next(iter(w._streams), None)
    ends = [m for m in api.received if m["type"] == "end_of_turn"] if api else []
    return out, (stream.counts if stream else {}), (stream.summary if stream else {}), ends


async def plain_run(script: list[tuple]) -> list[tuple]:
    return await collect(ScriptedSTT(script))


TURN = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like", 0.95),
    ev(0.30, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.55, T.INTERIM_TRANSCRIPT, "I'd like to check my", 0.95),
    ev(0.80, T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance.", 0.95),
    ev(0.95, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ev(0.95, T.END_OF_SPEECH),
]
TWO_TURNS = [
    *TURN,
    ev(1.40, T.START_OF_SPEECH),
    ev(1.45, T.INTERIM_TRANSCRIPT, "and my", 0.95),
    ev(1.70, T.INTERIM_TRANSCRIPT, "and my savings", 0.95),
    ev(1.90, T.FINAL_TRANSCRIPT, "And my savings."),
    ev(1.90, T.END_OF_SPEECH),
]
GUESS_BALANCE = "i'd like to check my balance"
AT_CHECK = {"I'd like to check": GUESS_BALANCE}


def hello_v2(**kw: Any) -> dict:
    return {
        "key": "test",
        "v": 2,
        "agent": "default",
        "host": "livekit",
        "host_version": agents_version,
        "hookup": rabel.__version__,
        "vendor": "scripted/script-1",
        "min_chance": 0.7,
        "learn": True,
        **kw,
    }


# ------------------------------------------------------------------ the hello and the options


async def test_hello_carries_version_2_fields_and_options_change_from_the_next_stream() -> None:
    api = FakeAPI()
    w = wrapper(TURN, api, agent="front-desk", user="cust-42:a_b.c")
    await collect(w)
    assert api.hello == hello_v2(agent="front-desk", user="cust-42:a_b.c")
    w.update_options(agent="demo-bank", min_chance=0.9, learn=False, user=None)
    await collect(w)
    assert api.hello == hello_v2(agent="demo-bank", min_chance=0.9, learn=False)


@pytest.mark.parametrize(
    "kw",
    [
        {"agent": "Front Desk"},
        {"min_chance": "0.8"},
        {"answer_timeout": 0},
        {"learn": "no"},
        {"user": "Jane Smith"},
        {"report": None},
    ],
    ids=lambda kw: "-".join(kw),
)
def test_options_are_checked(kw: dict) -> None:
    with pytest.raises(ValueError):
        rabel.STT(ScriptedSTT(TURN), api_key="test", **kw)
    w = rabel.STT(ScriptedSTT(TURN), api_key="test", agent="demo-bank")
    with pytest.raises(ValueError) as e:
        w.update_options(**kw)
    assert w._agent == "demo-bank"
    if "user" in kw:  # the error does not repeat the value
        assert kw["user"] not in str(e.value)


# ------------------------------------------------------------------ the rules


@pytest.mark.parametrize("v", [1, 2], ids=["api-v1", "api-v2"])
async def test_guess_sent_once_beginnings_held_eager_equal_held(v: int) -> None:
    api = FakeAPI(answers=AT_CHECK, v=v)
    out, counts, summary, ends = await wrapped_run(TURN, api)
    # one preflight from the guess; the interim that is a beginning of it and the speech-to-text's own preflight
    # with the same words are held back; the final and end of speech pass
    assert out == [
        (T.START_OF_SPEECH, None),
        (T.INTERIM_TRANSCRIPT, "I'd like"),
        (T.INTERIM_TRANSCRIPT, "I'd like to check"),
        (T.PREFLIGHT_TRANSCRIPT, "I'd like to check my balance."),
        (T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
        (T.END_OF_SPEECH, None),
    ]
    assert counts["guess_sent"] == 1 and counts["interim_held_back"] == 1
    assert counts["eager_held_back"] == 1
    heard = [m for m in api.received if m["type"] == "heard"]
    assert all(set(m) == {"type", "text"} for m in heard), heard  # only the words
    # sent at 0.32 s (the words at 0.30 s, the answer 0.02 s later), the final at 0.95 s
    assert ends == [
        {
            "type": "end_of_turn",
            "text": "I'd like to check my balance.",
            "guess": "kept",
            "head_start_ms": 630,
        }
    ]
    assert summary == {"turns": 1, "guesses": 1, "kept": 1, "dropped": 0, "head_start_ms": 630}


@pytest.mark.parametrize(
    "api_kw, stt_kw, dropped",
    [
        pytest.param(
            {"chance": 0.75, "min_chance": 0.8}, {}, "guess_below_min_chance", id="under-ready"
        ),
        pytest.param({"send_chance": False}, {}, "guess_no_chance", id="no-chance"),
        pytest.param({"delay": 0.3}, {"answer_timeout": 0.1}, "guess_slow", id="slow"),
    ],
)
async def test_wrong_or_slow_guess_not_used(
    api_kw: dict, stt_kw: dict, dropped: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    out, counts, _, ends = await wrapped_run(TURN, FakeAPI(answers=AT_CHECK, **api_kw), **stt_kw)
    assert out == await plain_run(TURN)
    assert counts[dropped] == 1 and "guess_sent" not in counts
    assert [m["guess"] for m in ends] == ["none"]
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == 1, lines


async def test_contradicting_interim_passes_and_eager_replaces() -> None:
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "check my savings", 0.95),
        ev(0.55, T.PREFLIGHT_TRANSCRIPT, "Check my savings.", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Check my savings."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    out, counts, _, ends = await wrapped_run(
        script, FakeAPI(answers={"check my": "check my balance"})
    )
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
    assert [(m["guess"], "head_start_ms" in m) for m in ends] == [("dropped", False)]


@pytest.mark.parametrize(
    "final, re_sent", [("check my balance.", True), ("Check my balance.", False)]
)
async def test_final_equal_to_a_cancelled_guess_goes_first_as_an_early_start(
    final: str, re_sent: bool
) -> None:
    # LiveKit remembers the cancelled guess's text and starts no early reply at a final equal to it
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.INTERIM_TRANSCRIPT, "check my", 0.95),
        ev(0.30, T.INTERIM_TRANSCRIPT, "check my bills", 0.95),
        ev(0.50, T.INTERIM_TRANSCRIPT, "check my balance", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, final),
        ev(0.70, T.END_OF_SPEECH),
    ]
    out, counts, _, _ = await wrapped_run(script, FakeAPI(answers={"check my": "check my balance"}))
    tail = [(T.PREFLIGHT_TRANSCRIPT, final)] if re_sent else []
    assert out[-(len(tail) + 2) :] == [*tail, (T.FINAL_TRANSCRIPT, final), (T.END_OF_SPEECH, None)]
    assert counts.get("final_sent_as_early_start", 0) == len(tail) and counts["guess_sent"] == 1


# LiveKit starts its own early reply on BEFORE's final, which waits for its end of turn
BEFORE = [
    ev(0.00, T.START_OF_SPEECH),
    ev(0.05, T.INTERIM_TRANSCRIPT, "I'd like to check", 0.95),
    ev(0.30, T.FINAL_TRANSCRIPT, "I'd like to check my balance."),
    ev(0.30, T.END_OF_SPEECH),
]
AFTER = [
    ev(0.60, T.START_OF_SPEECH),
    ev(0.65, T.INTERIM_TRANSCRIPT, "and my", 0.95),
    ev(0.95, T.INTERIM_TRANSCRIPT, "and my savings", 0.95),
    ev(1.20, T.FINAL_TRANSCRIPT, "And my savings please."),
    ev(1.20, T.END_OF_SPEECH),
]


@pytest.mark.parametrize(
    "commit_at, sent, count",
    [
        pytest.param(None, False, "held_guess_not_sent_turn_over", id="livekit-reply-waits"),
        pytest.param(0.80, False, "held_guess_not_sent_commit", id="commit-after-the-guess"),
        pytest.param(0.45, True, None, id="commit-before-the-guess"),
    ],
)
async def test_guess_held_while_livekits_own_reply_waits(
    commit_at: float | None, sent: bool, count: str | None
) -> None:
    script = [*BEFORE, *AFTER]
    if commit_at is not None:
        script = sorted(
            [*script, ev(commit_at, COMMIT, "I'd like to check my balance.")], key=lambda e: e[0]
        )
    out, counts, _, _ = await wrapped_run(
        script, FakeAPI(answers={"and my": "and my savings please"})
    )
    guesses = [e for e in out if e[0] == T.PREFLIGHT_TRANSCRIPT]
    if sent:
        assert guesses == [(T.PREFLIGHT_TRANSCRIPT, "and my savings please.")]
    else:
        assert guesses == [] and counts["guess_held_while_livekit_waits"] == 1
        assert count is not None and counts[count] == 1
        assert out == [(typ, text) for _, typ, text, _ in [*BEFORE, *AFTER]]


async def test_no_guess_as_third_early_start_and_none_equal_to_words() -> None:
    script = [
        ev(0.00, T.START_OF_SPEECH),
        ev(0.05, T.PREFLIGHT_TRANSCRIPT, "yes", 0.95),
        ev(0.20, T.PREFLIGHT_TRANSCRIPT, "yes please", 0.95),
        ev(0.35, T.INTERIM_TRANSCRIPT, "yes please do", 0.95),
        ev(0.70, T.FINAL_TRANSCRIPT, "Yes, please do it."),
        ev(0.70, T.END_OF_SPEECH),
    ]
    api = FakeAPI(answers={"yes please do": "yes please do it", "yes": "yes"})
    out, counts, _, _ = await wrapped_run(script, api)
    assert counts["guess_third_start"] == 1 and counts["guess_equals_words"] == 1
    assert out == await plain_run(script)


# ------------------------------------------------------------------ fails open


@pytest.mark.parametrize(
    "case, lines_expected",
    [
        ("down", 1),
        ("disabled", 0),
        ("no-key", 1),
        ("slow-ready", 0),
        ("slow-guess", 0),
        ("refused-402", 1),
        ("refused-503", 1),
        ("garbage", 2),
    ],
)
async def test_fails_open(case: str, lines_expected: int, caplog: pytest.LogCaptureFixture) -> None:
    plain = await plain_run(TURN)
    answers = {"I'd like": GUESS_BALANCE, **AT_CHECK}
    api: FakeAPI | None = {
        "down": None,
        "slow-ready": FakeAPI(answers=answers, ready_after=5.0),
        "slow-guess": FakeAPI(answers=answers, delay=2.0),
        "refused-402": FakeAPI(mode="refuse", status=402, error="no_seconds"),
        "refused-503": FakeAPI(mode="refuse", status=503, error="waking", retry_after=60),
        "garbage": FakeAPI(mode="garbage"),
    }.get(case, FakeAPI(answers=answers))
    kw: dict[str, Any] = {"enabled": False} if case == "disabled" else {}
    if case == "no-key":
        kw["api_key"] = ""
    caplog.set_level(logging.WARNING, logger="livekit.plugins.rabel")
    settle = {"slow-guess": 2.5, "down": 3.0}.get(case, 0.5)
    out, *_ = await wrapped_run(TURN, api, settle=settle, **kw)
    assert out == plain, case
    lines = [r.getMessage() for r in caplog.records if r.name == "livekit.plugins.rabel"]
    assert len(lines) == lines_expected, lines


async def test_reopens_after_503_with_backoff_and_not_after_400() -> None:
    api = FakeAPI(plan=["refuse", "ok"], status=503, error="waking", retry_after=0)
    await wrapped_run(TWO_TURNS, api)
    at = [t - api.connected_at[0] for t in api.connected_at]
    assert len(at) == 2 and at[1] == pytest.approx(1.0, abs=0.01), at
    assert ("end_of_turn", "And my savings.") in [(m["type"], m.get("text")) for m in api.received]
    api = FakeAPI(plan=["refuse"], status=400, error="bad_message", retry_after=0)
    await wrapped_run(TWO_TURNS, api)
    assert len(api.connected_at) == 1


# ------------------------------------------------------------------ the report and the two log lines


@pytest.mark.parametrize("report", [True, False], ids=["report-on", "report-off"])
async def test_log_lines_written_with_the_report_on_or_off_and_carry_no_words(
    report: bool, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.rabel")
    answers = {**AT_CHECK, "and my": "and my savings please"}
    api = FakeAPI(answers=answers)
    out, _, summary, ends = await wrapped_run(TWO_TURNS, api, report=report)
    if report:
        assert [m["guess"] for m in ends] == ["kept", "dropped"]
    else:  # the API is told nothing of the guesses
        assert ends == [
            {"type": "end_of_turn", "text": "I'd like to check my balance."},
            {"type": "end_of_turn", "text": "And my savings."},
        ]
    recs = [r for r in caplog.records if r.name == "livekit.plugins.rabel"]
    kept = [r for r in recs if r.getMessage() == "Bell's guess kept"]
    assert len(kept) == 1 and kept[0].levelno == logging.DEBUG and kept[0].head_start_ms == 630
    totals = [r for r in recs if "head start in all" in r.getMessage()]
    assert len(totals) == 1 and totals[0].levelno == logging.INFO
    assert (
        totals[0].getMessage()
        == "Bell: 2 turns, 2 guesses, 1 kept, 1 dropped, 630 ms head start in all"
    )
    assert summary == {"turns": 2, "guesses": 2, "kept": 1, "dropped": 1, "head_start_ms": 630}
    said: set[str] = set()
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


async def test_report_switched_off_at_once() -> None:
    api = FakeAPI(answers=AT_CHECK)
    w = wrapper(TWO_TURNS, api)
    stream = w.stream()
    await asyncio.sleep(0.2)
    stream.push_frame(
        rtc.AudioFrame(
            data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
        )
    )

    async def read() -> None:
        async for _ in stream:
            pass

    reader = asyncio.create_task(read())
    await asyncio.sleep(1.2)  # the first turn has ended; the second has not begun
    w.update_options(report=False)
    await asyncio.sleep(1.0)
    stream.end_input()
    await asyncio.wait_for(reader, 5)
    await stream.aclose()
    ends = [m for m in api.received if m["type"] == "end_of_turn"]
    assert [m.get("guess") for m in ends] == ["kept", None]
