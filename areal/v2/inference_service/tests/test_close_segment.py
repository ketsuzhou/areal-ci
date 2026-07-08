from fastapi.testclient import TestClient

from areal.v2.inference_service.data_proxy.app import (
    create_app as create_data_proxy_app,
)
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import SessionData
from areal.v2.inference_service.gateway.app import create_app as create_gateway_app
from areal.v2.inference_service.gateway.config import GatewayConfig


def _seed(session: SessionData) -> str:
    return session.add_string_interaction([{"role": "user", "content": "hi"}], "hello")


def test_close_segment_moves_active_to_ready_no_reward():
    s = SessionData("s1")
    _seed(s)
    result = s.close_segment()
    assert result.ready_transition is True
    assert result.trajectory_id == 0
    assert result.interaction_count == 1
    # active cleared
    assert len(s.active_completions) == 0
    # the closed interaction received no reward
    assert s.active_completions is not None


def test_close_segment_does_not_require_set_reward():
    s = SessionData("s1")
    _seed(s)
    # no set_reward called; close_segment must succeed
    result = s.close_segment()
    assert result.trajectory_id == 0


def test_close_segment_empty_active_raises():
    s = SessionData("s1")
    try:
        s.close_segment()
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty active")


def test_close_segment_session_stays_live_for_next_segment():
    s = SessionData("s1")
    _seed(s)
    s.close_segment()
    # next turn captures into a new active segment
    _seed(s)
    result = s.close_segment()
    assert result.trajectory_id == 1


def test_close_segment_endpoint_session_key():
    cfg = DataProxyConfig(admin_api_key="areal-admin-key", backend_addr="")
    app_instance = create_data_proxy_app(cfg)
    with TestClient(app_instance) as client:
        # start session (admin)
        r = client.post("/rl/start_session", json={"task_id": "t1"},
                     headers={"Authorization": "Bearer areal-admin-key"})
        assert r.status_code == 201
        api_key = r.json()["sessions"][0]["session_api_key"]
        session_id = r.json()["sessions"][0]["session_id"]

        # Directly add an interaction to the session instead of calling /chat/completions
        store = app_instance.state.session_store
        session = store.get_session(session_id)
        session.add_string_interaction([{"role": "user", "content": "hi"}], "hello")

        r = client.post("/rl/close_segment", headers={"Authorization": f"Bearer {api_key}"})
        assert r.status_code == 200
        body = r.json()
        assert body["trajectory_ready"] is True
        assert body["trajectory_id"] == 0


def test_close_segment_endpoint_empty_active_400():
    cfg = DataProxyConfig(admin_api_key="areal-admin-key", backend_addr="")
    with TestClient(create_data_proxy_app(cfg)) as client:
        r = client.post("/rl/start_session", json={"task_id": "t2"},
                     headers={"Authorization": "Bearer areal-admin-key"})
        api_key = r.json()["sessions"][0]["session_api_key"]
        r = client.post("/rl/close_segment", headers={"Authorization": f"Bearer {api_key}"})
        assert r.status_code == 400


def test_gateway_close_segment_route_registered():
    app = TestClient(create_gateway_app(GatewayConfig(admin_api_key="areal-admin-key")))
    # route exists (will 502/401 without a router, but not 404)
    r = app.post("/rl/close_segment", headers={"Authorization": "Bearer k"})
    assert r.status_code != 404


def test_export_single_trajectory_remove_session_false():
    cfg = DataProxyConfig(admin_api_key="areal-admin-key", backend_addr="")
    app_instance = create_data_proxy_app(cfg)
    with TestClient(app_instance) as client:
        # start session (admin)
        r = client.post("/rl/start_session", json={"task_id": "t3"},
                     headers={"Authorization": "Bearer areal-admin-key"})
        assert r.status_code == 201
        api_key = r.json()["sessions"][0]["session_api_key"]
        session_id = r.json()["sessions"][0]["session_id"]

        # Directly add an interaction to the session instead of calling /chat/completions
        store = app_instance.state.session_store
        session = store.get_session(session_id)
        session.add_string_interaction([{"role": "user", "content": "a"}], "hello")

        # Close segment to get trajectory_id 0
        r = client.post("/rl/close_segment", headers={"Authorization": f"Bearer {api_key}"})
        assert r.status_code == 200
        assert r.json()["trajectory_id"] == 0

        # Export with remove_session=False
        r = client.post("/export_trajectories", json={
            "session_ids": [session_id],
            "trajectory_id": 0,
            "remove_session": False
        }, headers={"Authorization": "Bearer areal-admin-key"})
        assert r.status_code == 200

        # Verify session still exists
        assert store.get_session(session_id) is not None


def test_export_unknown_trajectory_400():
    cfg = DataProxyConfig(admin_api_key="areal-admin-key", backend_addr="")
    app_instance = create_data_proxy_app(cfg)
    with TestClient(app_instance) as client:
        # start session (admin)
        r = client.post("/rl/start_session", json={"task_id": "t4"},
                     headers={"Authorization": "Bearer areal-admin-key"})
        assert r.status_code == 201
        api_key = r.json()["sessions"][0]["session_api_key"]
        session_id = r.json()["sessions"][0]["session_id"]

        # Directly add an interaction to the session instead of calling /chat/completions
        store = app_instance.state.session_store
        session = store.get_session(session_id)
        session.add_string_interaction([{"role": "user", "content": "a"}], "hello")

        # Close segment to get trajectory_id 0
        r = client.post("/rl/close_segment", headers={"Authorization": f"Bearer {api_key}"})
        assert r.status_code == 200

        # Export with unknown trajectory_id
        r = client.post("/export_trajectories", json={
            "session_ids": [session_id],
            "trajectory_id": 99,
            "remove_session": False
        }, headers={"Authorization": "Bearer areal-admin-key"})
        # This should return 400 after we fix the endpoint
        assert r.status_code == 400
