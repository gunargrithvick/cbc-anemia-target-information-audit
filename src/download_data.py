"""
Fetch the four NHANES cycles this project uses.

The files are official NCHS public-use files. Public-use availability does not
remove the NCHS data-use conditions: use them for statistical analysis or
reporting only, do not attempt to identify participants, and do not link them
with individually identifiable data. See the NCHS Data User Agreement:
https://www.cdc.gov/nchs/policy/data-user-agreement.html
Run:  python download_data.py            fetch whatever is missing or damaged
      python download_data.py --force    fetch all eight again, regardless

Why the download is checked and not just attempted
  urlretrieve wrote straight to the final filename and the only test applied
  afterwards was that the name existed. A connection dropped halfway therefore
  left a short file that every later run treated as complete, and pandas reads
  a truncated transport file without raising: cutting P_CBC.XPT in half yields
  6,875 rows and a UserWarning, so the loss would have arrived in the results
  as a smaller cohort rather than as an error. Downloads now go to a .part file,
  are checked against the server's Content-Length and the SAS transport
  structure, and are only then renamed into place. The structure check reads the
  observation length the file declares for itself and verifies the byte count is
  a whole number of observations plus blank padding; a cut that lands on an
  80-byte boundary AND keeps that alignment is invisible to structure, so the
  eight expected sizes and observation counts are pinned in EXPECT as well - see
  `problem`. Every run prints each file's observation count, so a file that
  shrinks says so on the line where it is reported.

Why four cycles
  2017-March 2020 is the primary cohort: it carries the survey design used for
  every weighted estimate. The three earlier cycles exist so the leakage ladder
  can be replicated on data the models were never fitted on, which is the only
  way to show the ladder is a property of the CBC and not of one sample.

A warning that matters for anyone reusing this
  The weight variable is NOT the same across cycles. The 2-year cycles carry
  WTMEC2YR; the pre-pandemic file carries WTMECPRP, which already covers 3.2
  years. SDMVSTRA and SDMVPSU are also cycle-specific: stratum 145 in 2011-12
  is not stratum 145 in 2015-16. So the cycles must not be pooled and then
  handed to one variance estimator. weighted.py stays inside 2017-2020 for
  exactly that reason, and the earlier cycles are used unweighted, as
  replication samples.
"""

import argparse
import math
import os
import struct
import sys
import urllib.request

from paths import DATA

if hasattr(sys.stdout, "reconfigure"):      # absent when stdout is wrapped
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

TIMEOUT = 120            # seconds per read; without it a stalled socket hangs
RECORD = 80              # SAS transport is a stream of fixed 80-byte records
MIN_BYTES = 200_000      # the smallest of the eight real files is ~1 MB

# First record of a transport file: "...LIBRARY HEADER RECORD..." in V5,
# "...LIBV8   HEADER RECORD..." in V8. The common prefix covers both.
XPT_MAGIC = b"HEADER RECORD*******LIB"

# The other three header records, by the token that names them. V8 spells them
# MEMBV8 / NAMSTV8 / OBSV8, so match the shortest unambiguous prefix of each.
HDR = b"HEADER RECORD*******"
MEMBER_TOKENS = (b"MEMBER", b"MEMBV8")
NAMESTR_TOKENS = (b"NAMESTR", b"NAMSTV8")
OBS_TOKENS = (b"OBS ", b"OBSV8")
SCAN_RECORDS = 16        # the header block is 8 records; 16 is slack, not hope

# cycle key -> (label, url year folder, cbc file, demo file, MEC weight column)
CYCLES = {
    "2011-2012": ("G", "2011", "CBC_G.XPT", "DEMO_G.XPT", "WTMEC2YR"),
    "2013-2014": ("H", "2013", "CBC_H.XPT", "DEMO_H.XPT", "WTMEC2YR"),
    "2015-2016": ("I", "2015", "CBC_I.XPT", "DEMO_I.XPT", "WTMEC2YR"),
    "2017-2020": ("P", "2017", "P_CBC.XPT", "P_DEMO.XPT", "WTMECPRP"),
}

PRIMARY = "2017-2020"

# What each file is, exactly: (bytes, observations, variables). Measured from the
# eight CDC files this study was run on, and the four DEMO observation counts are
# the published NHANES interviewed-sample sizes - 9,756 for 2011-12, 10,175 for
# 2013-14, 9,971 for 2015-16, 15,560 for 2017-March 2020 - so the table is
# checkable against the survey documentation rather than only against this disk.
#
# The point of pinning them is that structure alone cannot catch every
# truncation. A cut that drops whole 80-byte records and lands back on the same
# residue class modulo the observation length leaves a file that is internally
# consistent and simply shorter: for P_CBC.XPT that is one aligned cut in eleven,
# and cutting it exactly in half - the very example this module's docstring uses -
# is one of them. An expected size is the only thing that catches those, so it is
# recorded here instead of the check being described as stronger than it is.
#
# If CDC re-releases a file these numbers stop matching and every run says so
# loudly. That is the intended behaviour: a changed input invalidates the cohort
# counts printed in the README, so it needs a person, not a silent acceptance.
EXPECT = {
    "P_CBC.XPT":  (2_427_760, 13_772, 22),
    "P_DEMO.XPT": (3_614_720, 15_560, 29),
    "CBC_G.XPT":  (1_508_320,  8_956, 21),
    "DEMO_G.XPT": (3_753_760,  9_756, 48),
    "CBC_H.XPT":  (1_586_640,  9_422, 21),
    "DEMO_H.XPT": (3_833_200, 10_175, 47),
    "CBC_I.XPT":  (1_543_440,  9_165, 21),
    "DEMO_I.XPT": (3_756_480,  9_971, 47),
}


def cycle_files(cycle):
    """(cbc_path, demo_path) for one cycle key, as they land in data/."""
    _, _, cbc, demo, _ = CYCLES[cycle]
    return DATA / cbc, DATA / demo


def weight_col(cycle):
    return CYCLES[cycle][4]


def urls():
    """Every (cycle, destination, url) triple, primary cycle first."""
    order = [PRIMARY] + [c for c in CYCLES if c != PRIMARY]
    out = []
    for cycle in order:
        _, year, cbc, demo, _ = CYCLES[cycle]
        for name in (cbc, demo):
            # CDC serves lower-case .xpt; we store the documented upper-case name
            url = f"{BASE}/{year}/DataFiles/{name.replace('.XPT', '.xpt')}"
            out.append((cycle, DATA / name, url))
    return out


def _find_header(head, tokens):
    """Index of the first 80-byte record naming one of `tokens`, or None."""
    for i in range(len(head) // RECORD):
        rec = head[i * RECORD:(i + 1) * RECORD]
        if rec.startswith(HDR) and any(t in rec[20:41] for t in tokens):
            return i
    return None


def layout(path):
    """
    Read a transport file's own declared geometry. Returns (info, reason).

    Exactly one of the two is None. On success `info` carries the observation
    length the file declares for itself and the number of whole observations the
    byte count therefore contains:

        {"variables", "namestr_bytes", "record_length", "observations",
         "data_bytes", "padding_bytes"}

    Only the first 4 KB and the last 80 bytes are read, so this costs nothing and
    can run on every file on every invocation.

    The geometry comes from the V5/V8 header block, which a transport file is
    required to carry before any data:

      MEMBER  header  columns 76-78 give the namestr length, 140 or 136
      NAMESTR header  columns 55-58 give the variable count
      namestrs        one per variable, each declaring its own field width at
                      byte offset 4 as a big-endian short; they sum to the
                      observation length
      OBS     header  sits immediately after the namestr block, padded up to the
                      next 80-byte boundary, and everything after it is data

    That last line is the point of the whole function: knowing the observation
    length turns "how many bytes are there" into "is that a whole number of
    observations", which is the question a truncation actually fails.
    """
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(SCAN_RECORDS * RECORD)

    mem_i = _find_header(head, MEMBER_TOKENS)
    ns_i = _find_header(head, NAMESTR_TOKENS)
    if mem_i is None or ns_i is None:
        return None, "carries no MEMBER/NAMESTR header block"
    try:
        namestr_bytes = int(head[mem_i * RECORD + 75:mem_i * RECORD + 78])
        nvars = int(head[ns_i * RECORD + 54:ns_i * RECORD + 58])
    except ValueError:
        return None, "has a MEMBER/NAMESTR header whose length fields are not numeric"
    if not 0 < nvars <= 10_000 or namestr_bytes not in (136, 140):
        return None, (f"declares {nvars} variables of {namestr_bytes}-byte "
                      f"namestrs, which is not a transport file's geometry")

    ns_start = (ns_i + 1) * RECORD
    ns_block = math.ceil(nvars * namestr_bytes / RECORD) * RECORD
    obs_off = ns_start + ns_block
    with open(path, "rb") as fh:
        fh.seek(ns_start)
        namestrs = fh.read(ns_block)
        obs_hdr = fh.read(RECORD)
    if len(namestrs) < nvars * namestr_bytes or not (
            obs_hdr.startswith(HDR) and any(t in obs_hdr[20:41] for t in OBS_TOKENS)):
        return None, (f"has no OBS header where its own namestr block ends "
                      f"(byte {obs_off:,}), so it is cut inside the header")

    record_length = sum(
        struct.unpack(">h", namestrs[v * namestr_bytes + 4:
                                     v * namestr_bytes + 6])[0]
        for v in range(nvars))
    if record_length <= 0:
        return None, "declares a non-positive observation length"

    data_bytes = size - (obs_off + RECORD)
    if data_bytes < 0:
        return None, "ends before its own data section starts"
    pad = data_bytes % record_length
    return {"variables": nvars, "namestr_bytes": namestr_bytes,
            "record_length": record_length,
            "observations": data_bytes // record_length,
            "data_bytes": data_bytes, "padding_bytes": pad}, None


def problem(path):
    """
    Why `path` is not a usable NHANES transport file, or None if it is.

    Six tests, cheapest first. None of them needs the network, so run_all.py can
    use this to decide whether the download stage may be skipped instead of
    trusting the filename the way it used to.

    The first three - present, not tiny, a whole number of 80-byte records,
    begins with a library header - were the whole check, and they are not enough.
    A dropped connection cuts at whatever byte the socket stopped at, and roughly
    one cut in eighty lands on a record boundary; such a file passed all three and
    was accepted. That is the failure this module's own docstring demonstrates,
    6,875 rows instead of 13,772, so the check was passing exactly the file it
    was written to catch.

    Tests four and five read the geometry the file declares for itself (see
    `layout`) and ask whether the byte count agrees:

      4. the data section must be a whole number of observations, give or take
         the final partial record;
      5. that remainder is blank padding, so it must be shorter than one record
         AND be nothing but spaces. All eight real files satisfy this - their
         padding is 0 to 64 bytes of 0x20 - while a truncation usually leaves
         live data there.

    Usually, not always, and that is why there is a sixth. A cut that drops whole
    records and happens to leave the byte count in the same residue class modulo
    the observation length produces a file that is internally consistent and
    simply shorter - one aligned cut in eleven for P_CBC.XPT, and the half-file
    cut above is one of them. Structure cannot see that, because there is nothing
    structurally wrong with it. Test six compares the size and observation count
    against EXPECT, which is what makes the answer exact rather than probabilistic
    and is the test that actually catches the motivating example.

    So five of the six are for a file of unknown provenance and the sixth is for
    these eight known files. `fetch` additionally checks the server's
    Content-Length before anything is renamed into place, so a short download
    never reaches this function in the first place.

    What none of the six sees is a byte changed in place: the size, the geometry
    and the observation count all survive it. Catching that needs a checksum of
    the CDC files, which this project does not publish, so it is not claimed. The
    six answer "is this the right amount of data", not "is every value intact".
    """
    if not path.exists():
        return "absent"
    size = path.stat().st_size
    if size < MIN_BYTES:
        return f"only {size:,} bytes; the real files are megabytes"
    if size % RECORD:
        return (f"{size:,} bytes is not a whole number of {RECORD}-byte "
                f"records, so it was cut mid-record")
    with open(path, "rb") as fh:
        if fh.read(len(XPT_MAGIC)) != XPT_MAGIC:
            return "does not begin with a SAS transport library header"

    info, why = layout(path)
    if why:
        return why
    pad = info["padding_bytes"]
    if pad >= RECORD:
        return (f"{info['data_bytes']:,} data bytes leave {pad:,} over "
                f"{info['observations']:,} observations of "
                f"{info['record_length']} bytes, more than the {RECORD} bytes of "
                f"padding a complete file can end with, so records are missing")
    if pad:
        with open(path, "rb") as fh:
            fh.seek(size - pad)
            tail = fh.read(pad)
        if tail.strip(b" ") != b"":
            return (f"its last {pad} bytes are a partial observation carrying "
                    f"live data rather than blank padding, so it is truncated")

    want = EXPECT.get(path.name)
    if want is not None:
        got = (size, info["observations"], info["variables"])
        if got != want:
            return (f"is {got[0]:,} bytes / {got[1]:,} observations / {got[2]} "
                    f"variables where this study's file is {want[0]:,} / "
                    f"{want[1]:,} / {want[2]}; either it is damaged or CDC has "
                    f"re-released it, and the second case needs EXPECT and the "
                    f"cohort counts in README.md updated together")
    return None


def fetch(url, dest):
    """
    Download one file atomically. Raises on anything short of a clean copy.

    The bytes land in <dest>.part and are renamed only after they pass the
    length and structure checks, so an interrupted or rejected download leaves
    dest either untouched or absent - never half written.
    """
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        declared = resp.headers.get("Content-Length")
        with open(part, "wb") as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)

    got = part.stat().st_size
    try:
        if declared is not None and got != int(declared):
            raise OSError(f"{dest.name}: server announced {int(declared):,} "
                          f"bytes, received {got:,}")
        why = problem(part)
        if why:
            raise OSError(f"{dest.name}: downloaded file {why}")
    except OSError:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return got


def _shape(path):
    """"n obs x n vars" for a valid file, for the have/get lines in main."""
    info, why = layout(path)
    return "" if why else (f"{info['observations']:>7,} obs x "
                           f"{info['variables']:>3} vars")


def main(force=False):
    os.makedirs(DATA, exist_ok=True)
    got = bad = 0
    for cycle, dest, url in urls():
        why = problem(dest)
        if why is None and not force:
            print(f"  have  {cycle}  {dest.name:14s} "
                  f"{dest.stat().st_size:>10,} bytes   {_shape(dest)}")
            continue
        # A file that exists but fails the structure check is refetched, not
        # accepted and not merely reported: leaving it in place is how a
        # truncated XPT used to become a silently smaller cohort.
        reason = "--force" if why is None else why
        print(f"  get   {cycle}  {dest.name:14s} <- {url}   ({reason})")
        try:
            size = fetch(url, dest)
        except Exception as exc:                        # noqa: BLE001
            print(f"        FAILED: {exc}")
            bad += 1
            continue
        print(f"        {size:,} bytes   {_shape(dest)}")
        got += 1

    n = len(urls())
    print(f"\n{n - bad} of {n} files present and structurally valid, "
          f"{got} downloaded this run")
    if bad:
        raise SystemExit(f"download_data: {bad} file(s) unusable; "
                         f"nothing downstream can be trusted until they are")

    print("\ncodebooks:")
    for cycle, (_, year, cbc, demo, wt) in CYCLES.items():
        stem_c, stem_d = cbc.replace(".XPT", ""), demo.replace(".XPT", "")
        tag = "  (primary, survey design)" if cycle == PRIMARY else ""
        print(f"  {cycle}  weight {wt}{tag}")
        print(f"    {BASE}/{year}/DataFiles/{stem_c}.htm")
        print(f"    {BASE}/{year}/DataFiles/{stem_d}.htm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="re-download every file even if it is already valid")
    main(force=ap.parse_args().force)
