import os
from fastmcp import FastMCP
from kubernetes import client, config

mcp = FastMCP("Kubernetes Cluster Inspector")

# Use in-cluster config when running inside K8s, fallback to kubeconfig locally
try:
    config.load_incluster_config()
except config.ConfigException:
    config.load_kube_config()


@mcp.tool()
def list_nodes() -> dict:
    """Lists all cluster nodes with roles, architecture, OS, capacity, allocatable headroom, and pressure flags."""
    v1 = client.CoreV1Api()
    nodes = v1.list_node().items
    results = []

    for n in nodes:
        # Conditions map (Ready, MemoryPressure, DiskPressure, PIDPressure)
        conditions = {c.type: c.status for c in (n.status.conditions or [])}

        # Roles extraction
        roles = [
            k.split("/")[-1]
            for k in (n.metadata.labels or {}).keys()
            if "node-role.kubernetes.io" in k
        ]
        if not roles:
            roles = ["control-plane", "master"] if "k3s" in n.metadata.name else ["worker"]

        results.append({
            "name": n.metadata.name,
            "roles": roles,
            "kubelet_version": n.status.node_info.kubelet_version,
            "os_image": n.status.node_info.os_image,
            "architecture": n.status.node_info.architecture,
            "kernel_version": n.status.node_info.kernel_version,
            "container_runtime": n.status.node_info.container_runtime_version,
            "conditions": conditions,
            "capacity": {
                "cpu": n.status.capacity.get("cpu"),
                "memory": n.status.capacity.get("memory"),
                "pods": n.status.capacity.get("pods"),
            },
            "allocatable": {
                "cpu": n.status.allocatable.get("cpu"),
                "memory": n.status.allocatable.get("memory"),
                "ephemeral_storage": n.status.allocatable.get("ephemeral-storage"),
            },
        })

    return {"nodes": results}


@mcp.tool()
def list_pods(namespace: str = "default") -> dict:
    """Lists pods in a namespace with container statuses, restart counts, and phase."""
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=namespace).items
    results = []
    for p in pods:
        restarts = sum(cs.restart_count for cs in (p.status.container_statuses or []))
        ready = all(cs.ready for cs in (p.status.container_statuses or [])) if p.status.container_statuses else False
        results.append({
            "name": p.metadata.name,
            "phase": p.status.phase,
            "ready": ready,
            "restart_count": restarts,
            "node_name": p.spec.node_name,
        })
    return {"pods": results}


@mcp.tool()
def get_pod_logs(pod_name: str, namespace: str = "default", tail_lines: int = 100) -> dict:
    """Fetches the most recent logs from a given pod."""
    v1 = client.CoreV1Api()
    try:
        logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=tail_lines)
        return {"pod": pod_name, "logs": logs}
    except Exception as e:
        return {"pod": pod_name, "error": str(e)}


@mcp.tool()
def get_warning_events(namespace: str = "default") -> dict:
    """Retrieves warning events in a namespace with timestamps, reasons, and target resources."""
    v1 = client.CoreV1Api()
    events = v1.list_namespaced_event(namespace=namespace).items
    warnings = [e for e in events if e.type == "Warning"]
    return {
        "warnings": [
            {
                "reason": e.reason,
                "message": e.message,
                "object": f"{e.involved_object.kind}/{e.involved_object.name}",
                "count": e.count,
                "last_seen": str(e.last_timestamp),
            }
            for e in warnings
        ]
    }


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)