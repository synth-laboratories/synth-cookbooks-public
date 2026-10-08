"""Managed Research SDK quickstart: start one hosted run and read its evidence.

Faithful mirror of https://docs.usesynth.ai/managed-research/sdk-quickstart
(the "Start a one-off run" + "Wait and inspect evidence" sections).

Setup:
    uv add "synth-ai[research]==0.11.2"   # or: pip install "synth-ai[research]==0.11.2"
    export SYNTH_API_KEY="sk_..."

Run:
    python quickstart.py

Objectives / experiments / milestones are managed by the runtime — you start a run
and read its evidence; you don't create or step them by hand. See
https://docs.usesynth.ai/managed-research/upgrading if you're on 0.11.1.
"""

from synth_ai import SynthClient


def main() -> None:
    client = SynthClient()

    # Start a one-off directed-effort run on the lite runbook.
    run = client.research.runs.start(
        "Review the project context and propose the smallest high-impact improvement.",
        host_kind="daytona",
        work_mode="directed_effort",
        providers=[{"provider": "openrouter"}],
        runbook="lite",
    )
    print("project:", run.project_id)
    print("run:", run.run_id)

    # Wait for the run to finish, then read back the evidence it left behind.
    result = run.wait(timeout=60 * 60, poll_interval=15)
    print("state:", result.public_state.value)
    print("stop reason:", result.stop_reason_message or result.stop_reason)

    for artifact in run.list_artifacts():
        print(
            artifact.get("artifact_id"),
            artifact.get("artifact_type"),
            artifact.get("title"),
        )


if __name__ == "__main__":
    main()
