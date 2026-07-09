from __future__ import annotations

import argparse
import contextlib
import html
import io
import os
import pickle
import re
import subprocess
import sys
import zipfile
from collections.abc import Generator, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import altair
import junitparser
import pandas
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TOKEN: str | None = None

# Latest substantial change in the output format of tests.yaml
# Disregard workflow runs older than this date.
CUTOFF = pandas.Timestamp("2026-06-04", tz="UTC")

# Columns kept for each cached artifact. This is exactly what the report needs;
# everything else (raw XML, job metadata, ...) is dropped as soon as possible.
COLUMNS = ["test", "date", "suite", "file", "html_url", "status", "message"]

# Mapping between a symbol (pass, fail, skip) and a color
COLORS = {
    "✓": "#acf2a5",
    "x": "#f2a5a5",
    "s": "#f2ef8f",
}


def get_token() -> str:
    """Read the GitHub API token from the GITHUB_TOKEN environment variable,
    falling back to the gh CLI if the variable is not set.
    """
    if token := os.environ.get("GITHUB_TOKEN"):
        return token
    try:
        proc = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "Failed to find a GitHub Token. Either set the GITHUB_TOKEN "
            "environment variable or log into the gh CLI (`gh auth login`)."
        ) from None
    return proc.stdout.strip()


def cache_name(repo: str) -> str:
    """Name of the local cache pickle, e.g. test_report.dask__distributed.pickle"""
    return f"test_report.{repo.replace('/', '__')}.pickle"


@contextlib.contextmanager
def get_session() -> Generator[requests.Session]:
    retry_strategy = Retry(
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=0.2,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    with requests.Session() as session:
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        yield session


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--repo",
        default="dask/distributed",
        help="github repository",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="git branch",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=["push", "schedule"],
        help="github events",
    )
    parser.add_argument(
        "--max-days",
        "-d",
        type=int,
        default=90,
        help="Maximum number of days to look back from now",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help="Maximum number of workflow runs to show (and, unless --report-only, "
        "to fetch artifacts for)",
    )
    parser.add_argument(
        "--nfails",
        "-n",
        type=int,
        default=1,
        help="Show test if it failed more than this many times",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test_report.html",
        help="Output file name",
    )
    parser.add_argument("--title", "-t", default="Test Report", help="Report title")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate the report straight from the local cache, without "
        "downloading new artifacts, pruning, or updating the cache on disk",
    )
    parser.add_argument(
        "--prune-db",
        action="store_true",
        help="Drop from the local cache any artifacts older than --max-days before "
        "saving it back to disk",
    )
    return parser.parse_args(argv)


def get_from_github(
    url: str, params: dict[str, Any], session: requests.Session
) -> requests.Response:
    """
    Make an authenticated request to the GitHub REST API.
    """
    r = session.get(url, params=params, headers={"Authorization": f"token {TOKEN}"})
    r.raise_for_status()
    return r


def get_all_pages(
    url: str, key: str, params: dict[str, Any], session: requests.Session
) -> list[dict]:
    """Fetch every page of a paginated GitHub list endpoint and concatenate the
    ``key`` field of each page.
    """
    link_regex = re.compile(r'<([^>]*)>;\s*rel="([\w]*)\"')
    out: list[dict] = []
    next_url: str | None = url
    while next_url:
        r = get_from_github(next_url, params, session=session)
        out += r.json()[key]
        next_url = None
        if link_headers := r.headers.get("Link"):
            links = dict((rel, href) for href, rel in link_regex.findall(link_headers))
            next_url = links.get("next")
    return out


def get_test_runs(
    repo: str, branch: str, events: list[str], max_days: int, session: requests.Session
) -> list[dict]:
    """List the most recent completed "Tests" workflow runs still within the
    retention window (i.e. created no more than ``max_days`` ago and no earlier
    than ``CUTOFF``).
    """
    since = (pandas.Timestamp.now(tz="UTC") - pandas.Timedelta(days=max_days)).date()
    since = max(since, CUTOFF.date())

    runs = []
    for event in events:
        runs += get_all_pages(
            f"https://api.github.com/repos/{repo}/actions/runs",
            key="workflow_runs",
            params={
                "per_page": 100,
                "branch": branch,
                "event": event,
                "created": f">={since}",
            },
            session=session,
        )

    return [
        r
        for r in runs
        if (
            pandas.to_datetime(r["created_at"]) >= CUTOFF
            and r["status"] == "completed"
            and r["conclusion"] != "cancelled"
            and r["name"].lower() == "tests"
        )
    ]


def get_artifacts(run_id: int, repo: str, session: requests.Session) -> list[dict]:
    """List the JUnit XML artifacts of a workflow run."""
    artifacts = get_all_pages(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts",
        key="artifacts",
        params={"per_page": 100},
        session=session,
    )
    return [
        a
        for a in artifacts
        if not a["expired"]
        and a["name"] != "Event File"
        and "cluster_dumps" not in a["name"]
    ]


def get_job_urls(run: dict, session: requests.Session) -> dict[str, str]:
    """Map each artifact name to the html_url of the job that produced it.

    Job names are space-separated, e.g. ``ubuntu-latest py310 test-ci not ci1``,
    whereas the matching artifact is named after $TEST_ID (see the ``Set $TEST_ID``
    step in tests.yaml), which is dash-separated with the space removed from the
    partition name: ``ubuntu-latest-py310-test-ci-notci1``. dask/dask job names are
    the same, minus the trailing partition.
    """
    jobs = get_all_pages(
        run["jobs_url"], key="jobs", params={"per_page": 100}, session=session
    )
    return {
        job["name"].replace(" not ci1", " notci1").replace(" ", "-"): job["html_url"]
        for job in jobs
        if job["name"] != "Event File"
    }


def suite_from_name(name: str) -> str:
    """
    Get a test suite name from an artifact name. This is the artifact name
    minus the trailing pytest partition (dask/distributed only), so that the
    ci1 and notci1 partitions of the same test suite are shown on one row.
    """
    return re.sub(r"-(not)?ci1$", "", name)


def dataframe_from_jxml(run: Iterable) -> pandas.DataFrame:
    """
    Turn a parsed JXML into a pandas dataframe
    """
    fname = []
    tname = []
    status = []
    message = []
    for suite in run:
        for test in suite:
            fname.append(test.classname)
            tname.append(test.name)
            s = "✓"
            result = test.result

            if len(result) == 0:
                status.append(s)
                message.append("")
                continue
            result = result[0]
            m = result.message if result and hasattr(result, "message") else ""
            if isinstance(result, junitparser.Skipped):
                s = "s"
            else:
                # junitparser.Error, junitparser.Failure, or anything unexpected
                s = "x"
            status.append(s)
            message.append(html.escape(m))
    df = pandas.DataFrame(
        {"file": fname, "test": tname, "status": status, "message": message}
    )

    # There are sometimes duplicate tests in the report for some unknown reason.
    # If that is the case, concatenate the messages and prefer to show errors.
    def dedup(group):
        if len(group) > 1:
            if "message" in group.name:
                return group.str.cat(sep="")
            elif (group == "x").any(axis=0):
                return "x"
            else:
                return group.iloc[0]
        else:
            return group

    return df.groupby(["file", "test"], as_index=False).agg(dedup)


def download_artifact(
    a: dict, date: str, job_urls: dict[str, str], session: requests.Session
) -> pandas.DataFrame | None:
    """Download a single artifact, parse it, and reduce it to a dataframe holding
    just the ``COLUMNS`` needed by the report. The raw download and parsed XML are
    discarded immediately. Returns None if the artifact can't be used.
    """
    r = get_from_github(a["archive_download_url"], params={}, session=session)
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    try:
        xml = junitparser.JUnitXml.fromstring(zf.read(zf.filelist[0].filename))
    except Exception:
        # e.g. truncated XML from a job that hit the timeout
        return None

    df = dataframe_from_jxml(cast(Iterable, xml))
    if df.empty:
        return None

    html_url = job_urls.get(a["name"])
    if html_url is None:
        print(f"  WARNING: no job matches artifact {a['name']!r}; skipping")
        return None

    # Note: we tag every artifact with the workflow run timestamp rather than the
    # artifact timestamp, so that artifacts of the same run align on the same
    # trigger time.
    return df.assign(suite=suite_from_name(a["name"]), date=date, html_url=html_url)[
        COLUMNS
    ]


def download_new_artifacts(
    cache: dict[str, pandas.DataFrame],
    repo: str,
    branch: str,
    events: list[str],
    max_days: int,
    max_runs: int,
) -> None:
    """Download into ``cache`` (in place) any artifact of the most recent
    ``max_runs`` workflow runs that isn't already cached, keyed by its download url.
    """
    print("Getting list of workflow runs...")
    with get_session() as session:
        runs = get_test_runs(repo, branch, events, max_days, session)
        runs = sorted(runs, key=lambda r: r["created_at"])[-max_runs:]
        print(f"Fetching artifacts for the {len(runs)} most recent workflow runs")

        ndownloaded = 0
        for run in runs:
            new = [
                a
                for a in get_artifacts(run["id"], repo, session)
                if a["archive_download_url"] not in cache
            ]
            if not new:
                continue
            # Only fetch the jobs listing for runs that have something to download
            job_urls = get_job_urls(run, session)
            for a in new:
                df = download_artifact(a, run["created_at"], job_urls, session)
                if df is not None:
                    cache[a["archive_download_url"]] = df
                ndownloaded += 1
                if ndownloaded % 100 == 0:
                    print(f"{ndownloaded} downloaded...", flush=True)

    print(f"Downloaded {ndownloaded} new artifacts ({len(cache)} in cache)")


def load_cache(path: str) -> dict[str, pandas.DataFrame]:
    """Load the whole cache from ``path``, or start fresh if it's missing or
    unreadable (e.g. after a format change).
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
        assert isinstance(cache, dict)
    except Exception as e:
        print(f"Could not load cache {path} ({e!r}); starting fresh")
        return {}
    print(f"Loaded {len(cache)} artifacts from {path}")
    return cache


def dump_cache(path: str, cache: dict[str, pandas.DataFrame]) -> None:
    """Atomically write the whole cache to ``path``."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"Saved {len(cache)} artifacts to {path}")


def prune_cache(cache: dict[str, pandas.DataFrame], max_days: int) -> None:
    """Drop from ``cache`` (in place) every artifact older than the retention
    window (``max_days`` ago, but never earlier than ``CUTOFF``).
    """
    since = max(
        pandas.Timestamp.now(tz="UTC") - pandas.Timedelta(days=max_days), CUTOFF
    )
    stale = [
        url for url, df in cache.items() if pandas.to_datetime(df.date.iloc[0]) < since
    ]
    for url in stale:
        del cache[url]
    print(f"Pruned {len(stale)} artifacts older than {since.date()}")


def make_chart(name, df, times):
    # Create an aggregated form of the suite with overall pass rate
    # over the time in question.
    df_agg = (
        df[df.status != "x"]
        .groupby("suite")
        .size()
        .truediv(df.groupby("suite").size(), fill_value=0)
        .to_frame(name="Pass Rate")
        .reset_index()
    )

    # Create a grid with hover tooltip for error messages
    return altair.Chart(df).mark_rect(stroke="gray").encode(
        x=altair.X("date:O", scale=altair.Scale(domain=sorted(list(times)))),
        y=altair.Y("suite:N", title=None),
        href=altair.Href("html_url:N"),
        color=altair.Color(
            "status:N",
            scale=altair.Scale(
                domain=list(COLORS.keys()),
                range=list(COLORS.values()),
            ),
        ),
        tooltip=["suite:N", "date:O", "status:N", "message:N", "html_url:N"],
    ).properties(title=name) | altair.Chart(df_agg.assign(_="_")).mark_rect(
        stroke="gray"
    ).encode(
        y=altair.Y("suite:N", title=None, axis=altair.Axis(labels=False)),
        x=altair.X("_:N", title=None),
        color=altair.Color(
            "Pass Rate:Q",
            scale=altair.Scale(range=[COLORS["x"], COLORS["✓"]], domain=[0.0, 1.0]),
        ),
        tooltip=["suite:N", "Pass Rate:Q"],
    )


def build_report(
    cache: dict[str, pandas.DataFrame],
    repo: str,
    title: str,
    output: str,
    max_days: int,
    max_runs: int,
    nfails: int,
    argv: list[str] | None,
) -> None:
    since = max(
        pandas.Timestamp.now(tz="UTC") - pandas.Timedelta(days=max_days), CUTOFF
    )
    dfs = [df for df in cache.values() if pandas.to_datetime(df.date.iloc[0]) >= since]
    if not dfs:
        print("Nothing to report")
        with open(output, "w") as f:
            f.write(f"<html><body><h1>{repo} {title}</h1><p>No data</p></body></html>")
        return

    total = pandas.concat(dfs, axis=0, ignore_index=True)
    # Keep only the most recent `max_runs` workflow runs (one per trigger time)
    keep_dates = sorted(total.date.unique())[-max_runs:]
    total = total[total.date.isin(keep_dates)]

    # Note: we drop **all** tests which did not have at least <nfails> failures.
    # This is because, as nice as a block of green tests can be, there are
    # far too many tests to visualize at once, so we only want to look at
    # flaky tests. If the test suite has been doing well, this chart should
    # dwindle to nothing!
    grouped = (
        total.groupby([total.file, total.test])
        .filter(lambda g: (g.status == "x").sum() >= nfails)
        .reset_index()
        .assign(test=lambda df: df.file.str.cat(df.test, sep="."))
        .groupby("test")
    )
    overall = {name: grouped.get_group(name) for name in grouped.groups}

    # Get all the workflow run timestamps that we wound up with, which we can use
    # below to align the different groups.
    times: set = set()
    for df in overall.values():
        times.update(df.date.unique())

    print("Making chart...")
    altair.data_transformers.disable_max_rows()

    with ProcessPoolExecutor() as executor:
        jobs = [
            executor.submit(make_chart, name, df, times)
            for name, df in overall.items()
            if len(df)
        ]
        charts = [job.result() for job in jobs]

    if not charts:
        print("No flaky tests!")
        with open(output, "w") as f:
            f.write(
                f"<html><body><h1>{repo} {title}</h1>"
                "<p>No flaky tests 🎉</p></body></html>"
            )
        return

    # Concat the sub-charts and output to file
    chart = (
        altair.vconcat(*charts)
        .properties(
            title={
                "text": [f"{repo} {title}"],
                "subtitle": [" ".join(argv if argv is not None else sys.argv)],
            }
        )
        .configure_axis(labelLimit=1000)  # test names are long
        .configure_title(
            anchor="start",
            subtitleFont="monospace",
        )
        .resolve_scale(x="shared")  # enforce aligned x axes
    )

    chart.save(
        output,
        embed_options={
            "renderer": "svg",  # Makes the text searchable
            "loader": {"target": "_blank"},  # Open hrefs in a new window
        },
    )


def main(argv: list[str] | None = None) -> None:
    global TOKEN

    args = parse_args(argv)
    path = cache_name(args.repo)
    cache = load_cache(path)

    if not args.report_only:
        TOKEN = get_token()
        download_new_artifacts(
            cache,
            repo=args.repo,
            branch=args.branch,
            events=args.events,
            max_days=args.max_days,
            max_runs=args.max_runs,
        )
        if args.prune_db:
            prune_cache(cache, args.max_days)
        dump_cache(path, cache)

    build_report(
        cache,
        repo=args.repo,
        title=args.title,
        output=args.output,
        max_days=args.max_days,
        max_runs=args.max_runs,
        nfails=args.nfails,
        argv=argv,
    )


if __name__ == "__main__":
    main()
