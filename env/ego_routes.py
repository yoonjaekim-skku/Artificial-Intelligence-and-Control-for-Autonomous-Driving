"""Group ego routes by their actual edges, independent of route ID aliases."""
import xml.etree.ElementTree as ET


def group_unique_ego_routes(routes):
    """Return sorted (non-ring, ring) IDs, keeping one ID per edge sequence."""
    groups = ([], [])
    seen = set()
    # Prefer the Ring label for a Ring route that also has a shortest-path ID.
    for route_id in sorted(routes, key=lambda rid: (not rid.startswith("ego_r_"), rid)):
        if not route_id.startswith(("ego_d_", "ego_r_")):
            continue
        edges = tuple(routes[route_id])
        if edges in seen:
            continue
        seen.add(edges)
        uses_ring = any(edge.startswith("RING_") for edge in edges)
        groups[uses_ring].append(route_id)
    return tuple(sorted(groups[0])), tuple(sorted(groups[1]))


def read_ego_route_groups(path):
    routes = {node.attrib["id"]: node.attrib["edges"].split()
              for node in ET.parse(path).getroot().findall("route")}
    return group_unique_ego_routes(routes)
