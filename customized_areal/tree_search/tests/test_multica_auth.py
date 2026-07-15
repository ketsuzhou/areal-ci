import json
import os
import stat

import httpx
import pytest

from customized_areal.tree_search.agents import multica_auth
from customized_areal.tree_search.agents.multica_auth import (
    MulticaAuthError,
    load_saved_api_key,
    login,
    normalize_base_url,
    resolve_api_key,
    save_credentials,
)


def test_normalize_base_url_removes_trailing_slash():
    assert normalize_base_url(" https://multica.example.com/ ") == (
        "https://multica.example.com"
    )


@pytest.mark.parametrize(
    "base_url",
    ["", "multica.example.com", "ftp://multica.example.com", "https://x?q=1"],
)
def test_normalize_base_url_rejects_invalid_url(base_url):
    with pytest.raises(MulticaAuthError, match="absolute HTTP"):
        normalize_base_url(base_url)


def test_resolve_api_key_prefers_explicit_then_environment_then_saved(
    tmp_path, monkeypatch
):
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080/", "mul_saved", credentials_path=path)
    monkeypatch.setenv("MULTICA_API_KEY", "mul_env")

    assert (
        resolve_api_key("http://multica:8080", "mul_explicit", credentials_path=path)
        == "mul_explicit"
    )
    assert resolve_api_key("http://multica:8080", credentials_path=path) == "mul_env"

    monkeypatch.delenv("MULTICA_API_KEY")
    assert resolve_api_key("http://multica:8080", credentials_path=path) == "mul_saved"


def test_save_credentials_is_owner_only(tmp_path):
    path = tmp_path / "credentials.json"

    save_credentials("http://multica:8080", "mul_saved", credentials_path=path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == {
        "version": 1,
        "base_url": "http://multica:8080",
        "api_key": "mul_saved",
    }


def test_load_saved_api_key_rejects_base_url_mismatch(tmp_path):
    path = tmp_path / "credentials.json"
    save_credentials("http://multica-a:8080", "mul_saved", credentials_path=path)

    with pytest.raises(MulticaAuthError, match="different MultiCA server"):
        load_saved_api_key("http://multica-b:8080", credentials_path=path)


def test_load_saved_api_key_rejects_malformed_file_without_leaking_contents(tmp_path):
    path = tmp_path / "credentials.json"
    secret = "mul_must_not_leak"
    path.write_text(f'{{"api_key": "{secret}"')

    with pytest.raises(MulticaAuthError) as exc_info:
        load_saved_api_key("http://multica:8080", credentials_path=path)

    assert secret not in str(exc_info.value)


def test_load_saved_api_key_rejects_symlink(tmp_path):
    target = tmp_path / "real.json"
    target.write_text(
        json.dumps(
            {
                "version": 1,
                "base_url": "http://multica:8080",
                "api_key": "mul_saved",
            }
        )
    )
    path = tmp_path / "credentials.json"
    os.symlink(target, path)

    with pytest.raises(MulticaAuthError, match="regular file"):
        load_saved_api_key("http://multica:8080", credentials_path=path)


def test_load_saved_api_key_rejects_group_or_world_access(tmp_path):
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080", "mul_saved", credentials_path=path)
    path.chmod(0o644)

    with pytest.raises(MulticaAuthError, match="chmod 600"):
        load_saved_api_key("http://multica:8080", credentials_path=path)


def test_resolve_api_key_missing_credentials_points_to_login(tmp_path, monkeypatch):
    monkeypatch.delenv("MULTICA_API_KEY", raising=False)

    with pytest.raises(MulticaAuthError, match="multica_auth login"):
        resolve_api_key(
            "http://multica:8080", credentials_path=tmp_path / "missing.json"
        )


def test_login_exchanges_email_code_for_validated_pat(tmp_path):
    path = tmp_path / "credentials.json"
    seen = []

    def handler(request):
        seen.append(
            (
                request.url.path,
                request.headers.get("authorization"),
                json.loads(request.content) if request.content else None,
            )
        )
        if request.url.path == "/auth/send-code":
            return httpx.Response(200, json={"message": "Verification code sent"})
        if request.url.path == "/auth/verify-code":
            return httpx.Response(
                200,
                json={
                    "token": "jwt_temp",
                    "user": {"email": "user@example.com"},
                },
            )
        if request.url.path == "/api/tokens":
            return httpx.Response(201, json={"token": "mul_created"})
        if request.url.path == "/api/me":
            return httpx.Response(
                200, json={"email": "user@example.com", "name": "User"}
            )
        raise AssertionError(request.url.path)

    result = login(
        "http://multica:8080/",
        " User@Example.com ",
        " 123456 ",
        credentials_path=path,
        transport=httpx.MockTransport(handler),
        hostname="trainer-1",
    )

    assert result == {"email": "user@example.com", "name": "User"}
    assert seen == [
        ("/auth/send-code", None, {"email": "user@example.com"}),
        (
            "/auth/verify-code",
            None,
            {"email": "user@example.com", "code": "123456"},
        ),
        (
            "/api/tokens",
            "Bearer jwt_temp",
            {"name": "AReaL (trainer-1)", "expires_in_days": 90},
        ),
        ("/api/me", "Bearer mul_created", None),
    ]
    assert (
        load_saved_api_key("http://multica:8080", credentials_path=path)
        == "mul_created"
    )


@pytest.mark.parametrize(
    ("failure_path", "failure_status"),
    [
        ("/auth/verify-code", 400),
        ("/api/tokens", 500),
        ("/api/me", 401),
    ],
)
def test_login_failure_does_not_replace_existing_credentials(
    tmp_path, failure_path, failure_status
):
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080", "mul_existing", credentials_path=path)
    before = path.read_bytes()

    def handler(request):
        if request.url.path == failure_path:
            return httpx.Response(failure_status, json={"error": "secret details"})
        if request.url.path == "/auth/send-code":
            return httpx.Response(200, json={"message": "sent"})
        if request.url.path == "/auth/verify-code":
            return httpx.Response(200, json={"token": "jwt_temp", "user": {}})
        if request.url.path == "/api/tokens":
            return httpx.Response(201, json={"token": "mul_created"})
        if request.url.path == "/api/me":
            return httpx.Response(200, json={"email": "user@example.com"})
        raise AssertionError(request.url.path)

    with pytest.raises(MulticaAuthError) as exc_info:
        login(
            "http://multica:8080",
            "user@example.com",
            "123456",
            credentials_path=path,
            transport=httpx.MockTransport(handler),
            hostname="trainer-1",
        )

    assert str(failure_status) in str(exc_info.value)
    assert "secret details" not in str(exc_info.value)
    assert "jwt_temp" not in str(exc_info.value)
    assert "mul_created" not in str(exc_info.value)
    assert path.read_bytes() == before


def test_login_network_failure_is_sanitized_and_preserves_credentials(tmp_path):
    path = tmp_path / "credentials.json"
    save_credentials("http://multica:8080", "mul_existing", credentials_path=path)
    before = path.read_bytes()

    def handler(request):
        raise httpx.ConnectError(
            "connection failed with mul_sensitive", request=request
        )

    with pytest.raises(MulticaAuthError, match="network request failed") as exc_info:
        login(
            "http://multica:8080",
            "user@example.com",
            "123456",
            credentials_path=path,
            transport=httpx.MockTransport(handler),
        )

    assert "mul_sensitive" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert not hasattr(exc_info.value, "request")
    assert path.read_bytes() == before


def test_main_login_prompts_and_never_prints_pat(monkeypatch, capsys):
    seen = {}

    def fake_login(base_url, email, code):
        resolved_code = code() if callable(code) else code
        seen.update(base_url=base_url, email=email, code=resolved_code)
        return {"email": "user@example.com", "name": "User"}

    monkeypatch.setattr(multica_auth, "login", fake_login)
    monkeypatch.setattr("builtins.input", lambda prompt: "user@example.com")
    monkeypatch.setattr(multica_auth.getpass, "getpass", lambda prompt: "123456")

    assert multica_auth.main(["login", "--base-url", "http://multica:8080"]) == 0

    output = capsys.readouterr()
    assert seen == {
        "base_url": "http://multica:8080",
        "email": "user@example.com",
        "code": "123456",
    }
    assert "Authenticated as user@example.com" in output.out
    assert "123456" not in output.out + output.err
    assert "mul_" not in output.out + output.err
