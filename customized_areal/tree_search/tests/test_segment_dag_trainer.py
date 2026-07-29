import httpx
import orjson
import pytest

from customized_areal.tree_search.agents.segment_dag_trainer import (
    DataProxyTensorResolver,
)


def _shard_body(field_value):
    return orjson.dumps(
        {
            "type": "dataclass",
            "class_path": "areal.infra.rpc.rtensor.RTensor",
            "data": {"shard": {"shard_id": "x"}, "data": {"field_value": field_value}},
        }
    )


def test_resolve_multi_shard_fetches_each_field():
    tensor_ref = {
        "input_ids": {"shard_id": "shard-in", "node_addr": "h:1"},
        "attention_mask": {"shard_id": "shard-am", "node_addr": "h:1"},
    }
    requested = []

    def handler(request):
        requested.append((request.method, str(request.url)))
        path = str(request.url)
        if path == "http://x/data/shard-in":
            return httpx.Response(200, content=_shard_body("in"))
        if path == "http://x/data/shard-am":
            return httpx.Response(200, content=_shard_body("am"))
        return httpx.Response(404, content=b"not found")

    resolver = DataProxyTensorResolver(
        "http://x", _transport=httpx.MockTransport(handler)
    )
    out = resolver.resolve(tensor_ref)
    assert set(out.keys()) == {"input_ids", "attention_mask"}
    assert ("GET", "http://x/data/shard-in") in requested
    assert ("GET", "http://x/data/shard-am") in requested


def test_resolve_flattens_batched_tensor_field():
    """A ``[1, T]`` shard resolves to a flat length-T list.

    The data_proxy stores each field as the trainer-facing batched tensor, but
    ``_supernodes_to_batched_tensor_dict`` reads ``len()`` as the token count
    and scans ``loss_mask`` for the response span. Passing the batch dim
    through made that scan compare a whole row against ``1``.
    """
    import torch

    from areal.infra.rpc.serialization import serialize_value

    body = orjson.dumps(serialize_value(torch.tensor([[0, 0, 1, 1, 1]])))
    resolver = DataProxyTensorResolver(
        "http://x",
        _transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)),
    )
    out = resolver.resolve({"loss_mask": {"shard_id": "s"}})
    assert out["loss_mask"] == [0, 0, 1, 1, 1]


def test_resolve_flattens_nested_list_field():
    body = orjson.dumps([[0, 0, 1, 1, 1]])
    resolver = DataProxyTensorResolver(
        "http://x",
        _transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)),
    )
    out = resolver.resolve({"loss_mask": {"shard_id": "s"}})
    assert out["loss_mask"] == [0, 0, 1, 1, 1]


def test_resolve_leaves_flat_list_field_unchanged():
    body = orjson.dumps([10, 20, 30])
    resolver = DataProxyTensorResolver(
        "http://x",
        _transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)),
    )
    out = resolver.resolve({"input_ids": {"shard_id": "s"}})
    assert out["input_ids"] == [10, 20, 30]


def test_resolve_missing_shard_id_raises():
    resolver = DataProxyTensorResolver(
        "http://x", _transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    with pytest.raises(KeyError):
        resolver.resolve({"input_ids": {"node_addr": "h:1"}})


def test_resolve_404_raises_keyerror():
    resolver = DataProxyTensorResolver(
        "http://x", _transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    with pytest.raises(KeyError):
        resolver.resolve({"input_ids": {"shard_id": "ghost"}})
