#  Copyright 2025 Tuomas J. Lukka. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

if __name__ == "__main__":
    import uvicorn

    import sys

    print("Command line arguments:", " ".join(sys.argv))

    from grpo_server import grpo_service

    grpo_service.global_settings = grpo_service.Settings(_cli_parse_args=True)  # type: ignore

    # Run the FastAPI app using uvicorn
    uvicorn.run("grpo_server.grpo_service:app", host="0.0.0.0", port=8000, workers=1)
