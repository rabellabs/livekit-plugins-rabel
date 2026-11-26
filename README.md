# Rabel plugin for LiveKit Agents

Starts your [LiveKit](https://github.com/livekit/agents) voice agent's reply on [Rabel](https://rabellabs.com) Bell's
guess at the caller's final words, before the caller finishes.

Bell is a model behind Rabel's API that learns each agent's own callers from that agent's finished calls. It is in beta.
It works best on agents whose callers come for the same few things, like a bank's phone line or a booking desk.
Where callers talk freely, Bell has little to guess and mostly stays quiet, and the call runs as it would without it.

While the
caller speaks, it reads your speech-to-text's words and, once it has learned enough of that agent's calls, guesses how
the turn will end. The plugin wraps your speech-to-text and sends that guess to LiveKit as a preflight transcript, the
same signal Deepgram Flux's eager end of turn sends. LiveKit starts the LLM on it (preemptive generation), and its own
check keeps that reply only if the turn's final words match the guess. Your speech-to-text still decides where the turn
ends.

Bell runs on Rabel's servers: there is nothing to host and no GPU. It answers in about 3 ms to an agent in the same cloud
region (Google Cloud's us-east1) and about 72 ms from the US west coast, read from one Google Cloud machine in each (the
western one in Oregon) and not yet from LiveKit Cloud or any other network. One key holds up to 50 calls at once. Tested with LiveKit Agents 1.8.13. The numbers here come from
scripted calls to one bank, in English, with Deepgram Flux as the speech-to-text; the rest is under [Limits](#limits).

**What Rabel keeps.** Bell never receives audio. It takes nothing but the words from your speech-to-text and holds a
call's words only while that call is open. When the call closes, Bell learns from them for your agent and drops them.
What it learns stays with your agent and is never filed under a call or under a caller's name or number. Some of it is
your callers' own words, which can be a name or a number. You can list those at any time and delete any of them. The
rest is in a form that nobody but Rabel can match back to words. Deleting an agent erases what Bell learned for it.
Rabel's notice says it in full: [rabellabs.com/api-privacy](https://rabellabs.com/api-privacy).

Rabel Labs is not affiliated with, endorsed by or sponsored by LiveKit, Pipecat (Daily) or Deepgram; they are named only
to say what the package works with.

## Installation

```bash
pip install livekit-plugins-rabel
```

It needs `livekit-agents` 1.8.13 or later, before 2.0. Install your speech-to-text's plugin as you do today (for example
`livekit-plugins-deepgram`); this package does not bring one.

You need a Rabel API key, `demo-bank` included: see "Get an API key" in the [Rabel API docs](https://rabellabs.com/docs).
The price is on [Rabel's pricing page](https://rabellabs.com/pricing). Set the key in `RABEL_API_KEY`. Each speech
stream the plugin wraps (one for a call) keeps one conversation with the API open for the whole call, and the API meters
the seconds it is open, silence included.

## Usage

Wrap the speech-to-text you already use. This is the one changed line:

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import deepgram, rabel

session = AgentSession(
    stt=rabel.STT(deepgram.STTv2(model="flux-general-en", eager_eot_threshold=0.5), agent="front-desk"),
    turn_handling=TurnHandlingOptions(turn_detection="stt"),
    ...
)
```

A new agent starts with nothing learned, and Bell sends no guess until it has learned from enough of that agent's own
calls. The API's `ready` message, which opens each conversation, carries `calls_heard`: how many of the agent's calls
Bell has learned from. Your agent's status shows the same count under `v3` (see the Rabel API docs). To see it work at once, use
`agent="demo-bank"`: a ready-made agent that every account can use, already trained on a bank's callers (the
Gridspace-Stanford Harper Valley dataset of simulated bank calls, CC BY 4.0). A conversation that sounds like a call to
a bank gets guesses; talk on another topic rarely does. `demo-bank` never learns.

Any streaming speech-to-text with interim results can be wrapped. Bell goes with the speech-to-text's own early starts,
not instead of them: on turns where Bell has no guess, the events are the speech-to-text's, unchanged. Preemptive
generation is on by default in LiveKit, and Bell needs it on.

`turn_detection="stt"` is not required. Bell also works under LiveKit's own turn detection (its VAD and turn detector,
the default). There LiveKit keeps an early reply only when the final transcript is the same string as the guess, so
the plugin writes the guess the way your speech-to-text writes its finals.

Rabel's numbers were measured in a different setup from this snippet: LiveKit's own turn detector ends the turn,
preemptive generation is on and Flux's eager end of turn is off. That is
`rabel.STT(deepgram.STTv2(model="flux-general-en"), agent="front-desk")` with `turn_handling` left as it is.

## How a guess is handled

- A guess is used only within `answer_timeout` of the words it was made from, on the turn it answers, and at or above
  `min_chance` when you set one. A turn is sent to Bell only if its first words are: when the connection opens or
  reopens mid-turn, Bell starts with the next turn.
- One guess at a time is your agent's early reply. When the caller's words go another way, the reply started on that
  guess is dropped, and Bell may send a new guess in the same turn. A guess can also be the words heard so far: Bell's
  guess that the caller has finished.
- The plugin keeps within your session's limits for preemptive generation (`max_retries`, 3 by default, and
  `max_speech_duration`, 10 s by default). It reads them from the session, and from the running agent where the agent
  sets its own. A guess is never the last early start the session allows in a turn, and is never sent after the limit
  of speech.
- Where LiveKit starts no early reply at all, Bell sends no guess: with preemptive generation off, with
  `turn_detection="manual"`, or with a realtime model in place of the LLM.
- **With Flux's eager end of turn on**, Bell sends no guess once your speech-to-text has made its own early start in the
  turn, because the guess would cancel that start's reply or take a start the speech-to-text needs next. On a turn where
  Bell's guess took one of LiveKit's early starts, the speech-to-text's own early start is held back when none is left
  and a reply is running, because LiveKit would cancel that reply and start none.
- While Bell's guess is the running early start, the plugin holds back the interim transcripts and the speech-to-text's
  own preflight that agree with it, because LiveKit would otherwise cancel the reply. Anything that disagrees is passed
  and cancels the wrong reply. The final words still decide whether the reply is kept.
- While an early reply waits for LiveKit's end of turn (LiveKit's own, made on a final transcript, or one whose words a
  final transcript has matched), the plugin holds Bell's guess, which would replace that reply, and never sends it
  later. The plugin sees LiveKit's end of turn through the session's `stt_context_options` `forward_chat_context`, on by
  default. With it off, the plugin holds every guess that comes after such a reply begins to wait.

These rules make it rare for a guess to start a reply later than it would have started without Bell. Rare is not never.
In a replay of 300 scripted calls to one bank (1,607 turns), the reply started later than without Bell on 2 turns.

## Options

| Option | Default | |
|---|---|---|
| `agent` | `"default"` | Your agent's name: 1 to 64 lower-case letters, digits, `-` or `_`. Each agent learns from its own calls, and calls that share a name share what Bell learns: give each voice agent you run its own name and keep all of its calls under it. What Bell learns for it belongs to your account, so a new key keeps it. `"demo-bank"` is read-only. |
| `min_chance` | `None` | How sure Bell must be of a guess before it sends it, from `0.67` to `0.99`. Left at `None`, Bell uses its balanced setting. Set, every guess must also be at least that sure: fewer guesses, so fewer wrong ones (each costs your agent an extra call to its LLM) and fewer right ones. A value outside is clamped and one line is logged. |
| `answer_timeout` | `0.5` | Seconds after its words within which a guess is still used. |
| `learn` | `True` | `False` means the calls through this wrapper teach Bell nothing. |
| `user` | `None` | An id you choose for the person on the call: 1 to 128 letters, digits, `.`, `_`, `:` or `-`. Bell does not use it yet. Never a name, a number or an address. |
| `report` | `True` | Tells the API at each turn's end whether the guess was kept or dropped, with the head start of a kept one and no words, so your agent's status can show them. |
| `enabled` | `True` | `False` makes the wrapper a plain pass-through. |
| `api_key` | `RABEL_API_KEY` | Your Rabel API key. |
| `base_url` | `RABEL_API_URL`, else `https://api.rabellabs.com` | The API's address. |
| `http_session` | the job's own, else a new one | An `aiohttp.ClientSession` for the API's WebSocket. |

All but the last three can be changed with `update_options()`. `answer_timeout` and `report` take effect at once and
the others from the next speech stream. `user=None` stops sending the id, and `min_chance=None` goes back to the
balanced setting. An `agent`, `user`, `learn` or `report` not of the form above, a `min_chance` that is not a number
or `None`, or an `answer_timeout` not over 0 raises `ValueError`. Bell is sent the caller's words only, never the
agent's lines.

## Seeing Bell work

Set the `livekit.plugins.rabel` logger to `DEBUG`. Each guess sent logs "Bell: a guess sent as an early start" with its
chance, and each guess LiveKit kept logs "Bell's guess kept" with its head start in milliseconds. At `INFO`, one line
when the stream closes gives the call's turns, guesses, kept, dropped and head start in all. No line carries a word of
the call. LiveKit's own "using preemptive generation" line does not show Bell's guess was kept: LiveKit prints it for
every early reply it keeps.

The API's `GET /v1/agents/{agent}` shows the calls Bell has heard for your agent and, with `report` on,
`guesses_kept`, `guesses_dropped`, the head start and the agent's state (`learning`, `guessing` or `quiet`). With this plugin, where Bell stands for your
agent is under `v3`: `silent` until Bell has learned from enough of the agent's calls, then `guessing`. The plugin
reports your speech-to-text's model and provider as its own, so LiveKit's metrics and logs name your speech-to-text.

## Limits

- **The numbers come from one kind of call:** scripted calls to one bank, in English, with Flux's words. On freer speech we
  have only looked informally, and Bell mostly stayed quiet. No live deployment's callers and no other language have been
  measured, and the numbers were not read with Flux's eager end of turn on.
- **A later reply is rare, not impossible.** On some turns a guess can still start the reply later than it would have
  started without Bell.
- **Lower preemptive-generation limits mean fewer guesses.** The plugin never takes the last early start of a turn, so
  LiveKit always has one left for the caller's real words. With `max_retries` 2, Bell's guess can only be a turn's
  first early start. With `max_retries` 1, Bell sends none. Outside a LiveKit job (a script that drives the
  speech-to-text by itself) the plugin cannot read the session and assumes LiveKit's defaults.
- **Keep LiveKit's minimum interruption words at its default 0.** Set above 0 (`min_words` in `turn_handling`'s
  interruption options, `min_interruption_words` before), LiveKit counts a guess's words as the caller's. A "yeah" while
  your agent speaks can bring a guess long enough to interrupt or pause it.
- **Keep `forward_chat_context` on.** Off, Bell sends fewer guesses (above).
- **While a guess is live, your live caption shows the guess.** The user transcript LiveKit shows and sends to your
  front end carries Bell's guess, not the caller's words so far, until the words move past it or the final arrives. A
  front end cannot tell the two apart. Only `enabled=False` turns this off.
- **Turns ended by hand.** With `turn_detection="manual"` (push-to-talk) LiveKit starts no early reply, so Bell sends
  no guess and your speech-to-text's events pass unchanged. One case is left: an app that keeps turn detection
  automatic and also calls `session.commit_user_turn()` itself. When an early start arrived in the last half second,
  LiveKit does not wait for the final transcript and commits the text it holds, which is then Bell's guess. The same
  holds when the session closes within half a second of a guess. An app that ends turns by hand under automatic turn
  detection should set `enabled=False`.
- **It fails open.** If the API is slow, down, refuses or errors, the events are exactly your speech-to-text's, nothing
  is raised into the agent, and at most one line is logged for each kind of failure. With nothing listening, or no
  ready within 2 s of opening, it retries after 1, 2, 4, 8 and 16 s, then every 30 s. A 429 `key_at_once` or 529
  `out_of_capacity` retries the same way, or after the API's `retry_after_s` when that is longer (at most 300 s). A 400
  `bad_message`, 401 `bad_key`, 402 `no_seconds` or 404 `unknown_agent` ends Bell for that stream. Broken data from the
  API is ignored. A guess that comes late, under `min_chance` or after its turn has ended is dropped. With no API key, or
  `enabled=False`, no connection is opened. A conversation that closes cleanly after a minute or more (the API closes
  one after an hour) is reopened at once; any other is reopened as when the API is down. The tests run each of these
  cases against a local stand-in for versions 1, 2 and 3 of Rabel's API.
