"""
Example client for the inference API.

The pipeline's feature-selection step can pick a different set of columns on
every training run, so a request built from a fixed, hard-coded feature list
will usually be rejected. This script builds a valid request the same way any
real client should: read GET /health for the current feature names, then use
every one of them in /predict and /explain.

Usage (server must already be running — `python 03_mlops/serve.py`):
    python 03_mlops/example_client.py
"""
import json
import sys
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8000"


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=5) as resp:
        return json.load(resp)


def _post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        # A non-2xx response (e.g. /explain on a non-tree model) means the
        # server is reachable and answered — surface its error body rather
        # than treating this like a connection failure.
        return json.load(e)


def main() -> None:
    health = _get("/health")
    feature_names = health["expected_features"]
    print(f"/health reports {len(feature_names)} required features: {feature_names}")

    features = {name: 1.0 for name in feature_names}

    prediction = _post("/predict", {"features": features})
    print("/predict ->", prediction)

    explanation = _post("/explain", {"features": features})
    print("/explain ->", explanation)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError:
        print(
            f"Could not reach {BASE_URL} — start the API first with "
            "`python 03_mlops/serve.py`.",
            file=sys.stderr,
        )
        sys.exit(1)
