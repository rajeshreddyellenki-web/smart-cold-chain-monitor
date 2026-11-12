"""
Smart Cold Chain Monitor
------------------------
Reads temperature-sensor logs from refrigerated shipments and decides whether
the goods are still safe.

What it does for every shipment:
  1. Excursion detection   - finds periods outside the product's safe range,
                             with start time, duration and peak temperature.
  2. Rapid-change alerts   - flags sudden jumps (door left open, compressor fault).
  3. Breach prediction     - fits a trend to the latest readings and warns if the
                             temperature will leave the safe range soon.
  4. Shelf-life model      - Q10 rule: warmer storage uses up shelf life faster.
  5. Mean Kinetic Temp.    - the standard single number used for pharma/vaccines.
  6. Status + advice       - OK / WARNING / CRITICAL with a recommended action.

Input CSV columns:
  shipment_id, product, timestamp (YYYY-MM-DD HH:MM), temp_c, door_open (optional 0/1)

Usage:
  python cold_chain_monitor.py --simulate              # make demo data and analyse it
  python cold_chain_monitor.py sensor_log.csv          # analyse your own log
  python cold_chain_monitor.py sensor_log.csv --plot   # also draw graphs (needs matplotlib)
"""

import argparse
import csv
import math
import statistics
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

FMT = "%Y-%m-%d %H:%M"

# Safe range, reference storage temperature, shelf life at that temperature,
# Q10 factor (how many times faster it spoils per +10 C) and the longest
# out-of-range time (minutes) tolerated before the alert becomes CRITICAL.
PROFILES = {
    "vaccine": dict(low=2.0, high=8.0, ref=5.0, shelf_hours=720, q10=2.5, max_exc_min=30),
    "frozen": dict(low=-25.0, high=-15.0, ref=-20.0, shelf_hours=2160, q10=3.0, max_exc_min=60),
    "dairy": dict(low=1.0, high=5.0, ref=3.0, shelf_hours=336, q10=3.0, max_exc_min=45),
    "produce": dict(low=0.0, high=6.0, ref=3.0, shelf_hours=240, q10=2.0, max_exc_min=60),
}

RATE_LIMIT = 0.10        # C per minute treated as a suspicious jump
PREDICT_WINDOW = 8       # latest readings used for the trend line
PREDICT_HORIZON = 120    # warn if a breach is expected within this many minutes
DELTA_H_OVER_R = 10000   # K, standard value used for Mean Kinetic Temperature


# ------------------------------------------------------------------ data model
@dataclass
class Reading:
    shipment: str
    product: str
    ts: datetime
    temp: float
    door: bool


@dataclass
class Alert:
    shipment: str
    time: datetime
    severity: str   # WARNING or CRITICAL
    kind: str
    message: str


# ----------------------------------------------------------------------- input
def load_readings(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            product = row["product"].strip().lower()
            if product not in PROFILES:
                sys.exit(f"Unknown product '{product}'. Known: {', '.join(PROFILES)}")
            data[row["shipment_id"]].append(Reading(
                row["shipment_id"], product,
                datetime.strptime(row["timestamp"].strip(), FMT),
                float(row["temp_c"]),
                (row.get("door_open") or "0").strip().lower() in ("1", "true", "yes"),
            ))
    for readings in data.values():
        readings.sort(key=lambda r: r.ts)
    return data


# ------------------------------------------------------------------- analysis
def gaps_minutes(rs):
    """Minutes each reading represents (time until the next one)."""
    gaps = [(b.ts - a.ts).total_seconds() / 60 for a, b in zip(rs, rs[1:])]
    typical = statistics.median(gaps) if gaps else 15.0
    return gaps + [typical]


def find_excursions(rs, prof, gaps):
    """Return list of dicts describing each continuous out-of-range period."""
    result, cur = [], None
    for r, g in zip(rs, gaps):
        out = r.temp > prof["high"] or r.temp < prof["low"]
        dev = r.temp - prof["high"] if r.temp > prof["high"] else prof["low"] - r.temp
        if out:
            if cur is None:
                cur = dict(start=r.ts, minutes=0.0, peak=r.temp, dev=dev,
                           kind="HIGH" if r.temp > prof["high"] else "LOW", door=False)
            cur["minutes"] += g
            cur["door"] |= r.door
            if dev > cur["dev"]:
                cur["dev"], cur["peak"] = dev, r.temp
        elif cur:
            result.append(cur)
            cur = None
    if cur:
        result.append(cur)
    return result


def shelf_life_used(rs, prof, gaps):
    """Fraction of shelf life consumed, using the Q10 rule."""
    total = 0.0
    for r, g in zip(rs, gaps):
        rate = prof["q10"] ** ((r.temp - prof["ref"]) / 10.0)
        total += rate * g / (prof["shelf_hours"] * 60)
    return total


def ideal_life_used(prof, gaps):
    """Fraction that would be used if the goods stayed at the reference temperature."""
    return sum(gaps) / (prof["shelf_hours"] * 60)


def mean_kinetic_temperature(rs):
    k = [math.exp(-DELTA_H_OVER_R / (r.temp + 273.15)) for r in rs]
    return DELTA_H_OVER_R / -math.log(sum(k) / len(k)) - 273.15


def predict_breach(rs, prof):
    """Linear trend on the latest readings. Returns (kind, minutes) or None."""
    pts = rs[-PREDICT_WINDOW:]
    last = pts[-1].temp
    if len(pts) < 4 or not prof["low"] <= last <= prof["high"]:
        return None
    xs = [(p.ts - pts[0].ts).total_seconds() / 60 for p in pts]
    ys = [p.temp for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var  # C per minute
    if slope > 1e-4:
        mins = (prof["high"] - last) / slope
        kind = "HIGH"
    elif slope < -1e-4:
        mins = (last - prof["low"]) / -slope
        kind = "LOW"
    else:
        return None
    return (kind, mins, slope) if mins <= PREDICT_HORIZON else None


def analyse(shipment, rs):
    prof = PROFILES[rs[0].product]
    gaps = gaps_minutes(rs)
    alerts = []

    # 1. excursions
    excursions = find_excursions(rs, prof, gaps)
    for e in excursions:
        critical = e["minutes"] > prof["max_exc_min"] or e["dev"] > 5
        note = " (door open logged)" if e["door"] else ""
        alerts.append(Alert(
            shipment, e["start"], "CRITICAL" if critical else "WARNING", "EXCURSION",
            f"{e['kind']} excursion for {e['minutes']:.0f} min, peak {e['peak']:.1f} C "
            f"(safe {prof['low']:.0f} to {prof['high']:.0f} C){note}"))

    # 2. rapid changes
    for a, b, g in zip(rs, rs[1:], gaps):
        rate = (b.temp - a.temp) / g if g else 0
        if abs(rate) > RATE_LIMIT:
            alerts.append(Alert(
                shipment, b.ts, "WARNING", "RAPID_CHANGE",
                f"Temperature moved {b.temp - a.temp:+.1f} C in {g:.0f} min "
                f"({rate:+.2f} C/min) - check door seal or compressor"))

    # 3. predicted breach
    pred = predict_breach(rs, prof)
    if pred:
        kind, mins, slope = pred
        alerts.append(Alert(
            shipment, rs[-1].ts, "WARNING", "PREDICTED_BREACH",
            f"Trending {kind.lower()} at {slope * 60:+.2f} C/hour; "
            f"expected to leave safe range in about {mins:.0f} min"))

    # 4 and 5. shelf life and MKT
    used = shelf_life_used(rs, prof, gaps)
    ideal = ideal_life_used(prof, gaps)
    mkt = mean_kinetic_temperature(rs)
    remaining_h = max(0.0, (1 - used) * prof["shelf_hours"])
    mkt_ok = prof["low"] <= mkt <= prof["high"]
    if not mkt_ok:
        alerts.append(Alert(shipment, rs[-1].ts, "CRITICAL", "MKT",
                            f"Mean kinetic temperature {mkt:.1f} C is outside the safe range"))

    # 6. status
    severities = {a.severity for a in alerts}
    status = "CRITICAL" if "CRITICAL" in severities else "WARNING" if alerts else "OK"
    advice = {
        "OK": "Release normally.",
        "WARNING": "Inspect goods and check the refrigeration unit before release.",
        "CRITICAL": "Quarantine the load and have quality control decide before use or sale.",
    }[status]

    temps = [r.temp for r in rs]
    summary = dict(
        shipment=shipment, product=rs[0].product, readings=len(rs),
        hours=round(sum(gaps[:-1]) / 60, 1),
        min_c=round(min(temps), 1), max_c=round(max(temps), 1),
        mean_c=round(statistics.mean(temps), 1), mkt_c=round(mkt, 1),
        excursions=len(excursions),
        minutes_out_of_range=round(sum(e["minutes"] for e in excursions)),
        shelf_life_lost_pct=round(used * 100, 2),
        excess_loss_pct=round((used - ideal) * 100, 2),
        remaining_shelf_hours=round(remaining_h),
        status=status, advice=advice,
    )
    return summary, sorted(alerts, key=lambda a: a.time)


# ---------------------------------------------------------------------- output
def print_report(summaries, alerts_by_ship):
    print("\n" + "=" * 70)
    print("SMART COLD CHAIN MONITOR - SHIPMENT REPORT")
    print("=" * 70)
    for s in summaries:
        print(f"\n{s['shipment']}  [{s['product']}]  ->  {s['status']}")
        print(f"  Duration            : {s['hours']} h, {s['readings']} readings")
        print(f"  Temperature         : min {s['min_c']} / mean {s['mean_c']} / "
              f"max {s['max_c']} C   (MKT {s['mkt_c']} C)")
        print(f"  Out-of-range        : {s['excursions']} excursion(s), "
              f"{s['minutes_out_of_range']} min total")
        print(f"  Shelf life lost     : {s['shelf_life_lost_pct']}% "
              f"({s['excess_loss_pct']:+}% vs ideal storage), "
              f"about {s['remaining_shelf_hours']} h left")
        for a in alerts_by_ship[s["shipment"]]:
            print(f"  [{a.severity:8}] {a.time:%d %b %H:%M}  {a.kind}: {a.message}")
        print(f"  Action              : {s['advice']}")
    counts = {k: sum(1 for s in summaries if s["status"] == k) for k in ("OK", "WARNING", "CRITICAL")}
    print("\n" + "-" * 70)
    print(f"Fleet status: {counts['OK']} OK, {counts['WARNING']} WARNING, {counts['CRITICAL']} CRITICAL")


def save_csv(summaries, all_alerts, summary_path, alerts_path):
    with open(summary_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        wr.writeheader()
        wr.writerows(summaries)
    with open(alerts_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["shipment_id", "time", "severity", "type", "message"])
        for a in all_alerts:
            wr.writerow([a.shipment, a.time.strftime(FMT), a.severity, a.kind, a.message])


def plot(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping --plot)")
        return
    fig, axes = plt.subplots(len(data), 1, figsize=(9, 2.6 * len(data)), sharex=False)
    axes = [axes] if len(data) == 1 else axes
    for ax, (sid, rs) in zip(axes, data.items()):
        prof = PROFILES[rs[0].product]
        ax.axhspan(prof["low"], prof["high"], color="tab:green", alpha=0.15, label="safe range")
        ax.plot([r.ts for r in rs], [r.temp for r in rs], lw=1.4)
        ax.set_title(f"{sid} ({rs[0].product})", fontsize=10)
        ax.set_ylabel("C")
    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------- simulator
def simulate(path, seed=7):
    """Create demo data: one normal, one door-open, one drifting, one failure."""
    rng = random.Random(seed)
    start = datetime(2026, 11, 10, 6, 0)
    step, n = 15, 96  # 24 hours at 15-minute readings
    plans = [("SHP-001", "vaccine", "normal"), ("SHP-002", "vaccine", "door"),
             ("SHP-003", "dairy", "drift"), ("SHP-004", "frozen", "failure")]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["shipment_id", "product", "timestamp", "temp_c", "door_open"])
        for sid, prod, scenario in plans:
            p = PROFILES[prod]
            base = p["low"] + 1.0 if scenario == "drift" else p["ref"]
            temp = base
            for i in range(n):
                target, door = base, 0
                if scenario == "door" and 40 <= i <= 41:
                    target, door = base + 6, 1
                if scenario == "failure" and 30 <= i <= 69:
                    target = base + 14
                if scenario == "drift" and i >= 62:
                    temp = base + 0.08 * (i - 62) + rng.gauss(0, 0.1)
                else:
                    temp += (target - temp) * 0.35 + rng.gauss(0, 0.15)
                ts = start + timedelta(minutes=step * i)
                wr.writerow([sid, prod, ts.strftime(FMT), round(temp, 2), door])
    print(f"Demo sensor log written to {path}")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Smart Cold Chain Monitor")
    ap.add_argument("log_csv", nargs="?", default="sensor_log.csv")
    ap.add_argument("--simulate", action="store_true", help="create demo data first")
    ap.add_argument("--plot", action="store_true", help="draw graphs (needs matplotlib)")
    ap.add_argument("--summary-out", default="shipment_summary.csv")
    ap.add_argument("--alerts-out", default="alerts.csv")
    args = ap.parse_args()

    if args.simulate:
        simulate(args.log_csv)

    data = load_readings(args.log_csv)
    summaries, all_alerts, by_ship = [], [], {}
    for sid, rs in data.items():
        summary, alerts = analyse(sid, rs)
        summaries.append(summary)
        all_alerts.extend(alerts)
        by_ship[sid] = alerts

    print_report(summaries, by_ship)
    save_csv(summaries, all_alerts, args.summary_out, args.alerts_out)
    print(f"Saved {args.summary_out} and {args.alerts_out}")
    if args.plot:
        plot(data)


if __name__ == "__main__":
    main()
