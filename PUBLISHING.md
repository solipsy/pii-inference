# Publishing to PyPI

The package publishes as **`pii-inference`** (import name `privacy_filter`). Releases are
built and uploaded by GitHub Actions using **Trusted Publishing (OIDC)** — no API tokens
are stored anywhere. Tagging `vX.Y.Z` builds the wheels and publishes them.

## One-time setup

### 1. Confirm the name is claimed

`pii-inference` is currently free on PyPI. It becomes yours on the first successful
upload (or when you pre-register the trusted publisher below).

### 2. Create accounts

- A [PyPI](https://pypi.org/account/register/) account (and, for dry runs, a
  [TestPyPI](https://test.pypi.org/account/register/) account).
- Enable 2FA (required by PyPI).

### 3. Configure the PyPI trusted publisher

On PyPI → your account → **Publishing** → *Add a pending publisher* (this pre-registers
the project name before the first upload):

| Field | Value |
|---|---|
| PyPI Project Name | `pii-inference` |
| Owner | `solipsy` |
| Repository name | `pii-inference` |
| Workflow name | `wheels.yml` |
| Environment name | `pypi` |

This matches the `publish` job in `.github/workflows/wheels.yml`
(`environment: pypi`, `permissions: id-token: write`).

### 4. (Recommended) protect the `pypi` environment

GitHub → repo **Settings → Environments → New environment → `pypi`**. Optionally add a
required reviewer or restrict it to tag refs, so a publish can't happen by accident.

## Dry run on TestPyPI (optional but recommended)

Do one end-to-end test against TestPyPI first:

1. Add a matching **pending publisher on TestPyPI** (same fields, environment name
   `testpypi`).
2. Temporarily point the publish step at TestPyPI:
   ```yaml
   - uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
   ```
   (and set the job `environment: name: testpypi`).
3. Tag a pre-release, e.g. `v0.1.0rc1`, and push it. Verify the upload appears at
   <https://test.pypi.org/project/pii-inference/>, then install it:
   ```bash
   pip install -i https://test.pypi.org/simple/ pii-inference
   ```
4. Revert the `repository-url`/environment change.

## First release

1. Set the version in **both** places (they must match):
   - `pyproject.toml` → `[project].version`
   - `src/privacy_filter/__init__.py` → `__version__`
2. Commit, then tag and push:
   ```bash
   git commit -am "Release 0.1.0"
   git tag v0.1.0
   git push origin master --tags
   ```
3. The `wheels.yml` pipeline runs: **test → build wheels (Linux x86_64 + macOS arm64)
   → attach to GitHub Release → publish to PyPI**.
4. Confirm at <https://pypi.org/project/pii-inference/> and:
   ```bash
   pip install pii-inference
   python -c "import privacy_filter; print(privacy_filter.__version__)"
   ```

> **Note:** the git tag `v0.1.0` already exists from earlier wheel-build testing. For the
> first PyPI release, either delete/replace it (`git push origin :refs/tags/v0.1.0`) or
> just start at `v0.1.1` — a PyPI version can only be uploaded once (see below).

## Subsequent releases

Bump the version in both files, commit, tag `vX.Y.Z`, push. That's it.

## Notes & caveats

- **Versions are immutable.** PyPI refuses to re-upload a version that already exists.
  Always bump before tagging; never re-tag a shipped version.
- **Wheels only (no sdist).** We publish prebuilt `cp312-abi3` wheels for **Linux
  x86_64** and **macOS arm64**. There is intentionally no source distribution: an sdist
  would need the vendored `extern/privacy-filter.cpp` + `ggml` submodule sources bundled
  in, and `pip install` from it would require a full C++ toolchain. Users on other
  platforms should build [from source](docs/privacy-filter.md#install-from-source)
  instead. Add an sdist later if broader `pip install` coverage is needed.
- **Skip CI on the version-bump commit** if you add automated bumping later
  (`[skip ci]`), but the tag push must still trigger the workflow.
