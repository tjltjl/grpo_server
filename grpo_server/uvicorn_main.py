if __name__ == "__main__":
    import uvicorn

    import sys

    print("Command line arguments:", " ".join(sys.argv))

    from grpo_server import grpo_service

    grpo_service.global_settings = grpo_service.Settings(_cli_parse_args=True)  # type: ignore

    # Run the FastAPI app using uvicorn
    uvicorn.run("grpo_server.grpo_service:app", host="0.0.0.0", port=8000, workers=1)
