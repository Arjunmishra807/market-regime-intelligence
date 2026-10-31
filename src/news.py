"""Free RSS headlines + transparent lexicon sentiment/event tagging.
Headlines have no usable history, so news is used only in the live fusion layer
(never in the trained historical models). Heuristic - Not yet validated."""
from __future__ import annotations
import re, logging
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import numpy as np, pandas as pd, requests

log = logging.getLogger(__name__)
FEEDS = [("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
         ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html")]
POS = set("surge surges rally rallies gain gains rise rises jump jumps soar record beat beats strong growth "
          "optimism upgrade recover recovery rebound boost cooling easing".split())
NEG = set("plunge plunges fall falls drop drops slump selloff sell-off crash fear fears loss losses weak miss "
          "misses cut cuts layoffs downgrade recession default warns warning slide tumble tumbles crisis "
          "risk volatile turmoil panic".split())
EVENTS = {  # category: (keywords, severity 1-5)
    "inflation": (["inflation", "cpi", "pce"], 3),
    "interest rates": (["rate hike", "rate cut", "fed ", "yields", "treasury"], 3),
    "recession": (["recession", "gdp", "contraction", "slowdown"], 4),
    "geopolitical/war": (["war", "conflict", "attack", "sanction", "tariff", "missile", "geopolit"], 4),
    "banking stress": (["bank run", "bank failure", "banking", "credit crunch", "default"], 5),
    "liquidity": (["liquidity", "repo", "funding stress"], 4),
    "earnings": (["earnings", "revenue", "guidance", "profit"], 2),
    "layoffs": (["layoff", "job cuts", "unemployment"], 3),
    "regulation": (["regulat", "antitrust"], 2),
    "commodities": (["oil", "crude", "gold", "opec"], 2),
    "central bank": (["central bank", "ecb", "boj", "powell", "fomc"], 3),
}


def _score(t: str) -> float:
    w = re.findall(r"[a-z\-]+", t.lower()); p = sum(x in POS for x in w); n = sum(x in NEG for x in w)
    return (p - n) / (p + n + 1)


def _tag(t: str):
    tl = t.lower(); hits = [(c, s) for c, (kw, s) in EVENTS.items() if any(k in tl for k in kw)]
    return max(hits, key=lambda x: x[1]) if hits else ("general", 1)


def fetch_news(limit: int = 40):
    """Returns (df, summary dict | None, status)."""
    rows, now = [], pd.Timestamp.now(tz="UTC")
    for src, url in FEEDS:
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}); r.raise_for_status()
            for it in ET.fromstring(r.content).iter("item"):
                title = (it.findtext("title") or "").strip()
                try:
                    ts = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate")))
                    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                except Exception:
                    continue
                if title and ts <= now:   # never keep items stamped in the future
                    cat, sev = _tag(title)
                    rows.append(dict(time=ts, source=src, headline=title, sentiment=_score(title),
                                     event=cat, severity=sev, link=it.findtext("link")))
        except Exception as e:  # noqa
            log.warning("feed %s failed: %s", src, e)
    if not rows:
        return pd.DataFrame(), None, "UNAVAILABLE"
    df = pd.DataFrame(rows).drop_duplicates("headline").sort_values("time", ascending=False).head(limit)
    s = df.sentiment
    summ = dict(sentiment=float(s.mean()), volume=len(df), neg_ratio=float((s < 0).mean()),
                pos_ratio=float((s > 0).mean()), max_sev=int(df.severity.max()),
                mean_sev=float(df.severity.mean()))
    summ["risk"] = float(np.clip(50 - 35 * summ["sentiment"] + 12 * (summ["mean_sev"] - 1.5)
                                 + 15 * (summ["neg_ratio"] - 0.3), 0, 100))
    return df, summ, "DELAYED"
