from uuid import uuid4

from tests.unit.gateway.conftest import bearer


async def test_health_needs_no_credential(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_created_user_can_authenticate_on_the_very_next_request(client):
    created = await client.post("/api/v1/users", json={"display_name": "Demo"})
    assert created.status_code == 201
    token = created.json()["user_id"]

    me = await client.get("/api/v1/users/me", headers=bearer(token))

    assert me.status_code == 200
    assert me.json()["user_id"] == token
    assert me.json()["display_name"] == "Demo"
    assert me.json()["created_at"] == created.json()["created_at"]


async def test_user_can_be_created_without_a_body(client):
    assert (await client.post("/api/v1/users")).status_code == 201


async def test_unknown_field_and_oversized_name_are_rejected(client):
    unknown = await client.post("/api/v1/users", json={"role": "admin"})
    oversized = await client.post("/api/v1/users", json={"display_name": "x" * 81})

    assert unknown.status_code == 422
    assert unknown.json()["error"] == "validation_error"
    assert oversized.status_code == 422


async def test_missing_malformed_and_unknown_credentials_all_get_the_same_401(client):
    responses = [
        await client.get("/api/v1/users/me"),
        await client.get("/api/v1/users/me", headers={"Authorization": "Basic abc"}),
        await client.get("/api/v1/users/me", headers=bearer("not-a-uuid")),
        await client.get("/api/v1/users/me", headers=bearer(str(uuid4()))),
    ]

    for response in responses:
        assert response.status_code == 401
        assert response.json() == {"error": "invalid_credentials"}
        assert response.headers["www-authenticate"] == "Bearer"
