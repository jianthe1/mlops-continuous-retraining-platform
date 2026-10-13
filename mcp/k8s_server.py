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


if __name__ == "__main__":
    # Runs an SSE server listening on port 8000
    mcp.run(transport="sse", host="0.0.0.0", port=8000)