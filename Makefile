install:
	uv sync --frozen --all-groups

test:
	uv run pytest
