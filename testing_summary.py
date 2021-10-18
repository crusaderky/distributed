import io
import os
import pickle
import re
import shelve
import zipfile

import junitparser
import pandas
import requests


def get_from_github(url, params=None):
    """
    Make an authenticated request to the GitHub REST API.
    """
    print("GET", url)
    r = requests.get(
        url, params=params or {}, headers={"Authorization": f"token {TOKEN}"}
    )
    r.raise_for_status()
    return r


def maybe_get_next_page_path(response):
    """
    If a response is paginated, get the url for the next page.
    """
    link_regex = re.compile(r'<([^>]*)>;\s*rel="([\w]*)\"')
    link_headers = response.headers.get("Link")
    next_page_path = None
    if link_headers:
        links = {}
        matched = link_regex.findall(link_headers)
        for match in matched:
            links[match[1]] = match[0]
        next_page_path = links.get("next", None)

    return next_page_path


def get_artifact_listing(repo="crusaderky/distributed"):
    """
    Get a list of artifacts from GitHub actions
    """
    params = {"per_page": 1000}
    r = get_from_github(
        f"https://api.github.com/repos/{repo}/actions/artifacts", params=params
    )
    artifacts = r.json()["artifacts"]
    next_page = maybe_get_next_page_path(r)
    while next_page:
        r = get_from_github(next_page)
        artifacts = artifacts + r.json()["artifacts"]
        next_page = maybe_get_next_page_path(r)

    return artifacts


def dataframe_from_jxml(run, date):
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
            m = result.message if result and hasattr(result, "message") else ""
            if isinstance(result, junitparser.Error):
                s = "x"
            elif isinstance(result, junitparser.Failure):
                s = "x"
            elif isinstance(result, junitparser.Skipped):
                s = "s"
            status.append(s)
            message.append(m)
    df = pandas.DataFrame(
        {"file": fname, "test": tname, date: status, date + "-message": message}
    )
    return df.set_index(["file", "test"])


if __name__ == "__main__":
    TOKEN = os.environ.get("GITHUB_TOKEN")
    if not TOKEN:
        print(
            "Please add GitHub personal access token to your environment under GITHUB_TOKEN"
        )
        exit()

    # Get a listing of all artifacts
    print("Discovering all recent artifacts")
    artifacts = get_artifact_listing()
    # filter to the past week(?)
    print(f"Found {len(artifacts)} artifacts")
    artifacts = [
        a
        for a in artifacts
        if pandas.to_datetime(a["created_at"])
        > pandas.Timestamp.now(tz="UTC") - pandas.Timedelta(days=7)
    ]
    print(f"Downloading {len(artifacts)} artifacts")

    # Download the selected artifacts
    test_runs = {}
    with shelve.open("../testing_summary.shelf") as cache:
        for i, artifact in enumerate(artifacts):
            # Download the artifact and parse it
            url = artifact["archive_download_url"]
            try:
                content = cache[url]
            except KeyError:
                print(f"{i+1}/{len(artifacts)} ", end="")
                r = get_from_github(url)
                content = r.content
                cache[url] = content

            f = zipfile.ZipFile(io.BytesIO(content))
            run = junitparser.JUnitXml.fromstring(f.read(f.filelist[0].filename))

            # Insert into test runs dict
            name = artifact["name"]
            date = artifact["created_at"]
            test_runs.setdefault(name, {})
            test_runs[name][date] = run

    print("Generating test report")
    # Convert the JUnit data to dataframes
    dfs = {}
    for name, runs in test_runs.items():
        ll = [dataframe_from_jxml(suite, date) for date, suite in runs.items()]
        # Drop duplicated index values for now, figure out a better solution later
        ll = [df.loc[~df.index.duplicated()] for df in ll]
        dfs[name] = pandas.concat(ll, axis=1)

    with open("../testing_summary.pickle", "wb") as fh:
        pickle.dump(test_runs, fh)
