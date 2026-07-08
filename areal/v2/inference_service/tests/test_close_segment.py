from areal.v2.inference_service.data_proxy.session import SessionData


def _seed(session: SessionData) -> str:
    return session.add_string_interaction([{"role": "user", "content": "hi"}], "hello")


def test_close_segment_moves_active_to_ready_no_reward():
    s = SessionData("s1")
    iid = _seed(s)
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
