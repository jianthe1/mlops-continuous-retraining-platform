import os
from typing import Any, Dict, List
from fastmcp import FastMCP
from kubernetes import client, config

mcp = FastMCP("Kubernetes Cluster Inspector")


def get_k8s_client() -> client.CoreV1Api:
    """Loads in-cluster ServiceAccount token or falls back to local kubeconfig."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CoreV1Api()


@mcp.tool()
def list_pods(namespace: str = "default") -> List[Dict[str, Any]]:
    """List all pods in a given namespace along with status, node, and restart counts.

    Args:
        namespace: The Kubernetes namespace to inspect. Defaults to 'default'.
    """
    v1 = get_k8s_client()
    pods = v1.list_namespaced_pod(namespace=namespace)

    result = []
    for pod in pods.items:
        restarts = sum(
            cs.restart_count for cs in (pod.status.container_statuses or [])
        )
        result.append(
            {
                "name": pod.metadata.name,
                "status": pod.status.phase,
                "node": pod.spec.node_name,
                "restarts": restarts,
            }
        )
    return result


@mcp.tool()
def get_pod_logs(
    pod_name: str, namespace: str = "default", tail_lines: int = 50
) -> str:
    """Retrieve the recent container logs for a pod to diagnose crashes or errors.

    Args:
        pod_name: The name of the pod.
        namespace: The namespace where the pod lives. Defaults to 'default'.
        tail_lines: Number of lines to tail from the end of the log. Defaults
          to 50.
    """
    v1 = get_k8s_client()
    try:
        logs = v1.read_namespaced_pod_log(
            name=pod_name, namespace=namespace, tail_lines=tail_lines
        )
        return logs or "(No logs emitted)"
    except Exception as exc:
        return f"Error reading logs for {pod_name}: {str(exc)}"


@mcp.tool()
def list_nodes() -> List[Dict[str, Any]]:
    """List all nodes in the cluster with readiness state, kubelet version, and instance types."""
    v1 = get_k8s_client()
    nodes = v1.list_node()

    result = []
    for node in nodes.items:
        ready = next(
            (c.status for c in node.status.conditions if c.type == "Ready"),
            "Unknown",
        )
        result.append(
            {
                "name": node.metadata.name,
                "ready": ready == "True",
                "kubelet_version": node.status.node_info.kubelet_version,
                "instance_type": node.metadata.labels.get(
                    "node.kubernetes.io/instance-type", "N/A"
                ),
            }
        )
    return result


@mcp.tool()
def get_warning_events(namespace: str = "default") -> List[Dict[str, str]]:
    """List recent Warning events in a namespace (e.g., OOMKilled, BackOff, FailedScheduling).

    Args:
        namespace: The namespace to inspect. Defaults to 'default'.
    """
    v1 = get_k8s_client()
    events = v1.list_namespaced_event(namespace=namespace)

    result = []
    for event in events.items:
        if event.type == "Warning":
            result.append(
                {
                    "reason": event.reason,
                    "message": event.message,
                    "object": f"{event.involved_object.kind}/{event.involved_object.name}",
                    "count": str(event.count),
                }
            )
    return result

@mcp.tool()
def list_nodes() -> dict:
  """Lists all cluster nodes along with roles, OS/arch, allocatable capacity, and pressure conditions (MemoryPressure, DiskPressure, PIDPressure)."""
  v1 = client.CoreV1Api()
  nodes = v1.list_node().items
  results = []

  for n in nodes:
    # 1. Gather conditions (Ready, MemoryPressure, DiskPressure, PIDPressure)
    conditions = {c.type: c.status for c in (n.status.conditions or [])}

    # 2. Extract roles from labels
    roles = [
        k.split("/")[-1]
        for k in (n.metadata.labels or {}).keys()
        if "node-role.kubernetes.io" in k
    ]
    if not roles:
      roles = ["worker"]

    results.append({
        "name": n.metadata.name,
        "roles": roles,
        "kubelet_version": n.status.node_info.kubelet_version,
        "os": n.status.node_info.os_image,
        "architecture": n.status.node_info.architecture,
        "container_runtime": n.status.node_info.container_runtime_version,
        "conditions": conditions,  # e.g., {'Ready': 'True', 'MemoryPressure': 'False', ...}
        "capacity": {
            "cpu": n.status.capacity.get("cpu"),
            "memory": n.status.capacity.get("memory"),
            "pods": n.status.capacity.get("pods"),
        },
        "allocatable": {
            "cpu": n.status.allocatable.get("cpu"),
            "memory": n.status.allocatable.get("memory"),
        },
    })

  return {"nodes": results}

@mcp.tool()
def list_deployments(namespace: str = "default") -> dict:
  """Lists deployments with replica statuses (desired, updated, ready, available)."""
  apps_v1 = client.AppsV1Api()
  deps = apps_v1.list_namespaced_deployment(namespace=namespace).items
  return {
      "deployments": [{
          "name": d.metadata.name,
          "replicas_desired": d.spec.replicas,
          "replicas_ready": d.status.ready_replicas or 0,
          "replicas_updated": d.status.updated_replicas or 0,
          "replicas_available": d.status.available_replicas or 0,
          "images": [c.image for c in d.spec.template.spec.containers],
      } for d in deps]
  }


@mcp.tool()
def get_resource_events(namespace: str = "default", limit: int = 15) -> dict:
  """Fetches recent events in a namespace with timestamps, reasons, and target objects."""
  v1 = client.CoreV1Api()
  events = v1.list_namespaced_event(namespace=namespace).items
  events_sorted = sorted(
      events,
      key=lambda e: e.last_timestamp or e.event_time or "",
      reverse=True,
  )[:limit]

  return {
      "events": [{
          "type": e.type,
          "reason": e.reason,
          "object": f"{e.involved_object.kind}/{e.involved_object.name}",
          "message": e.message,
          "count": e.count,
          "last_seen": str(e.last_timestamp),
      } for e in events_sorted]
  }


if __name__ == "__main__":
    # Runs an SSE server listening on port 8000
    mcp.run(transport="sse", host="0.0.0.0", port=8000)