from __future__ import annotations

from cryptolab.features.catalog import (
    FEATURE_REGISTRY,
    FEATURE_SPECS,
)


def test_all_required_and_optional_dependencies_exist():
    feature_ids = set(
        FEATURE_REGISTRY.list_feature_ids()
    )

    for spec in FEATURE_SPECS:
        for dependency in (
            spec.all_input_features
        ):
            assert dependency in feature_ids, (
                f"{spec.feature_id} depends on "
                f"unknown feature {dependency}"
            )


def test_dependency_graph_with_optional_edges_is_acyclic():
    graph = {
        spec.feature_id: set(
            spec.all_input_features
        )
        for spec in FEATURE_SPECS
    }

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(
        node: str,
    ) -> None:
        if node in visited:
            return

        assert node not in visiting, (
            "Feature dependency cycle "
            f"detected at {node}"
        )

        visiting.add(
            node
        )

        for dependency in graph[
            node
        ]:
            visit(
                dependency
            )

        visiting.remove(
            node
        )

        visited.add(
            node
        )

    for feature_id in graph:
        visit(
            feature_id
        )
