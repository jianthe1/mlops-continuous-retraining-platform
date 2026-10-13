import os
from fastmcp import FastMCP
from kubernetes import client, config

mcp = FastMCP("Kubernetes Cluster Inspector")


def init_k8s():
  """Explicitly configures the Kubernetes API client for in-cluster execution."""
  try:
    config.load_incluster_config()
    print("Loaded in-cluster Kubernetes configuration.")
  except config.ConfigException:
    try:
      config.load_kube_config()
      print("Loaded local kubeconfig.")
    except Exception as e:
      print(f"Warning: Failed to load any kubeconfig: {e}")


# Initialize once at startup
init_k8s()


@mcp.tool()
def list_nodes() -> dict:
  """Lists all cluster nodes with roles, architecture, OS, capacity, allocatable headroom, and pressure flags."""
  try:
    v1 = client.CoreV1Api()
    nodes = v1.list_node().items
    results = []

    for n in nodes:
      conditions = {c.type: c.status for c in (n.status.conditions or [])}

      roles = [
          k.split("/")[-1]
          for k in (n.metadata.labels or {}).keys()
          if "node-role.kubernetes.io" in k
      ]
      if not roles:
        roles = (
            ["control-plane", "master"]
            if "k3s" in n.metadata.name
            else ["worker"]
        )

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
  except Exception as e:
    return {"error": f"Failed to list nodes: {str(e)}"}


@mcp.tool()
def list_pods(namespace: str = "default") -> dict:
  """Lists pods in a namespace with container statuses, restart counts, and phase."""
  try:
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=namespace).items
    results = []
    for p in pods:
      restarts = sum(
          cs.restart_count for cs in (p.status.container_statuses or [])
      )
      ready = (
          all(cs.ready for cs in (p.status.container_statuses or []))
          if p.status.container_statuses
          else False
      )
      results.append({
          "name": p.metadata.name,
          "phase": p.status.phase,
          "ready": ready,
          "restart_count": restarts,
          "node_name": p.spec.node_name,
      })
    return {"pods": results}
  except Exception as e:
    return {"error": f"Failed to list pods: {str(e)}"}


@mcp.tool()
def get_pod_logs(
    pod_name: str, namespace: str = "default", tail_lines: int = 100
) -> dict:
  """Fetches the most recent logs from a given pod."""
  v1 = client.CoreV1Api()
  try:
    logs = v1.read_namespaced_pod_log(
        name=pod_name, namespace=namespace, tail_lines=tail_lines
    )
    return {"pod": pod_name, "logs": logs}
  except Exception as e:
    return {"pod": pod_name, "error": str(e)}


@mcp.tool()
def get_warning_events(namespace: str = "default") -> dict:
  """Retrieves warning events in a namespace with timestamps, reasons, and target resources."""
  try:
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
  except Exception as e:
    return {"error": f"Failed to get events: {str(e)}"}


@mcp.tool()
def list_deployments(namespace: str = "default") -> dict:
  """Lists deployments with replica statuses."""
  try:
    apps_v1 = client.AppsV1Api()
    deps = apps_v1.list_namespaced_deployment(namespace=namespace).items
    return {
        "deployments": [{
            "name": d.metadata.name,
            "replicas_desired": d.spec.replicas,
            "replicas_ready": d.status.ready_replicas or 0,
            "replicas_available": d.status.available_replicas or 0,
        } for d in deps]
    }
  except Exception as e:
    return {"error": f"Failed to list deployments: {str(e)}"}


@mcp.tool()
def get_resource_events(namespace: str = "default", limit: int = 15) -> dict:
  """Fetches recent events in a namespace with timestamps, reasons, and target objects."""
  try:
    v1 = client.CoreV1Api()
    events = v1.list_namespaced_event(namespace=namespace).items
    events_sorted = sorted(
        events,
        key=lambda e: str(e.last_timestamp or e.event_time or ""),
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
  except Exception as e:
    return {"error": f"Failed to fetch resource events: {str(e)}"}


if __name__ == "__main__":
  mcp.run(transport="sse", host="0.0.0.0", port=8000)