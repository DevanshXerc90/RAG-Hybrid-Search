"""Environment for tests: dummy credentials must be set before app imports."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("DATABASE_NAME", "rag_example_schema")
os.environ.setdefault("DATABASE_USER", "postgres")
os.environ.setdefault("DATABASE_PASSWORD", "postgres")
os.environ.setdefault("DATABASE_URL", "localhost")
os.environ.setdefault("DATABASE_PORT", "5432")
