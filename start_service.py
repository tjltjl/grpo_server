import functools
import json
import re
import subprocess
import uvicorn


@functools.cache
def get_tailscale_ipv4():
    """Get the current host's tailscale ip4 address"""
    j = subprocess.check_output("tailscale status --json".split()).decode("utf-8")
    status = json.loads(j)

    ipv4s = [ip for ip in status["TailscaleIPs"] if re.match(r"^([0-9]+(\.|$)){4}", ip)]
    assert len(ipv4s) == 1
    return ipv4s[0]


def start_server():
    """Start the Uvicorn server"""
    uvicorn.run(
        "grpo_service:app",  # Path to your FastAPI app instance
        host=get_tailscale_ipv4(),
        port=9727,
        reload=True,  # Enable auto-reload on code changes
        log_level="trace",
    )


if __name__ == "__main__":
    start_server()
