from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from project_vitae.config import AppConfig
from project_vitae.graph import GraphState, build_graph
from project_vitae.models import SessionState
from project_vitae.session_lock import acquire_lock, release_lock


async def main() -> None:
    parser = argparse.ArgumentParser(description="ProjectVitae — agent-orchestrated resume builder")
    parser.add_argument("--config", default="userprofile/config.yaml", help="Path to config file")
    parser.add_argument("--session", default="default", help="Session name")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("urls", nargs="+", help="GitHub repository URLs")
    run_parser.add_argument("--jd", required=True, help="Path to job description file")

    setup_parser = subparsers.add_parser("setup", help="Run the TUI setup")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s %(name)s:%(lineno)d — %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        sys.exit(1)

    config = AppConfig.load(config_path)
    userprofile_dir = config_path.parent
    session_dir = userprofile_dir / "sessions" / args.session

    if args.command == "run":
        if not acquire_lock(session_dir):
            print(f"Session '{args.session}' is locked. Use --session with a different name or clear the lock.")
            sys.exit(1)

        try:
            jd_path = Path(args.jd)
            jd_text = jd_path.read_text(encoding="utf-8")

            state = GraphState(
                userprofile_dir=userprofile_dir,
                session_dir=session_dir,
                config=config,
                repo_urls=args.urls,
                session=SessionState(job_description=jd_text),
            )

            graph = build_graph()
            result = await graph.ainvoke(state)
            print("Pipeline complete!")
            if result.get("export_pdf"):
                print(f"PDF: {result['export_pdf']}")
        finally:
            release_lock(session_dir)

    elif args.command == "setup":
        from project_vitae.tui.app import run_tui
        await run_tui(config, userprofile_dir, session_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
