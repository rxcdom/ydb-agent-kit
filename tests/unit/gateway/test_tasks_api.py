from uuid import uuid4

from tests.unit.gateway.conftest import bearer

API = "/api/v1"


async def test_project_and_task_happy_path(client, token):
    headers = bearer(token)

    project = await client.post(
        f"{API}/projects", json={"name": "Boat", "description": "Hull"}, headers=headers
    )
    assert project.status_code == 201
    project_id = project.json()["project_id"]

    task = await client.post(
        f"{API}/tasks",
        json={"title": "Sand the hull", "project_id": project_id, "priority": "high"},
        headers=headers,
    )
    assert task.status_code == 201
    assert task.json()["status"] == "open"

    patched = await client.patch(
        f"{API}/tasks/{task.json()['task_id']}", json={"status": "done"}, headers=headers
    )
    assert patched.status_code == 200
    assert patched.json()["completed_at"] is not None

    listing = await client.get(f"{API}/tasks", params={"status": "done"}, headers=headers)
    assert listing.json()["total_count"] == 1
    assert listing.json()["has_more"] is False

    projects = await client.get(f"{API}/projects", headers=headers)
    assert projects.json()["projects"][0]["done_count"] == 1

    deleted = await client.delete(f"{API}/tasks/{task.json()['task_id']}", headers=headers)
    assert deleted.status_code == 204
    assert deleted.content == b""


async def test_task_endpoints_need_a_credential(client):
    for method, path in (("GET", "/tasks"), ("POST", "/tasks"), ("GET", "/projects")):
        response = await client.request(method, f"{API}{path}", json={"title": "x"})
        assert response.status_code == 401


async def test_another_owners_task_answers_like_a_missing_one(client, token):
    task = await client.post(f"{API}/tasks", json={"title": "Mine"}, headers=bearer(token))
    other = (await client.post(f"{API}/users", json={})).json()["user_id"]

    patched = await client.patch(
        f"{API}/tasks/{task.json()['task_id']}", json={"title": "Theirs"}, headers=bearer(other)
    )
    deleted = await client.delete(f"{API}/tasks/{task.json()['task_id']}", headers=bearer(other))
    missing = await client.delete(f"{API}/tasks/{uuid4()}", headers=bearer(other))

    assert patched.status_code == deleted.status_code == missing.status_code == 404
    assert patched.json() == missing.json() == {"error": "not_found"}


async def test_invalid_task_input_is_a_validation_error(client, token):
    headers = bearer(token)

    no_title = await client.post(f"{API}/tasks", json={"title": ""}, headers=headers)
    unknown_field = await client.post(
        f"{API}/tasks", json={"title": "x", "owner": "someone"}, headers=headers
    )
    inverted = await client.get(
        f"{API}/tasks", params={"date_from": "2026-09-02", "date_to": "2026-09-01"}, headers=headers
    )

    for response in (no_title, unknown_field, inverted):
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"
        assert response.json()["message"]
