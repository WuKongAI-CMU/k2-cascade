import math

from k2cascade.projector.sender_entropy import cluster, semantic_entropy


def test_cluster_and_entropy_without_nli():
    labels = cluster("Where?", ["north", "North.", "the north", "south", "east"], None)
    assert labels == [0, 0, 0, 1, 2]
    assert abs(semantic_entropy([0, 0, 0, 1, 2]) - (-(0.6 * math.log(0.6) + 2 * 0.2 * math.log(0.2)))) < 1e-9
    assert semantic_entropy([0, 0, 0]) == 0.0
