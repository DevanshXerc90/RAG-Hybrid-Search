"""Smoke tests: the application and evaluation assets import and wire up."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_app_imports_and_exposes_routes():
    from app import app

    paths = {route.path for route in app.routes}
    assert "/" in paths
    assert "/chat" in paths


def test_product_model_schema():
    from models.product import Product

    columns = Product.__table__.columns.keys()
    for expected in ("id", "name", "content", "embedding"):
        assert expected in columns


def test_postgres_searcher_modes_exist():
    from services.postgres_searcher import PostgresSearcher

    assert hasattr(PostgresSearcher, "search")
    assert hasattr(PostgresSearcher, "search_and_embed")


def test_eval_queries_are_labelled():
    data = json.loads(
        (REPO_ROOT / "scripts" / "eval_queries.json").read_text(encoding="utf-8")
    )
    queries = data["queries"]
    assert len(queries) == 30
    for item in queries:
        assert item["query"].strip()
        assert item["expected_ids"]
        assert all(1 <= product_id <= 30 for product_id in item["expected_ids"])
