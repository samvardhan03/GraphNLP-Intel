"""CLI: graphnlp run --input docs.csv --domain finance [--output output.html]"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:
    # Fallback: provide a minimal CLI without typer
    def main():
        print("typer is required for CLI. Install with: pip install typer")
        sys.exit(1)
else:
    cli = typer.Typer(
        name="graphnlp",
        help="GraphNLP Intel — Hybrid Graph-NLP Intelligence Platform CLI",
        add_completion=False,
    )

    @cli.command()
    def run(
        input: str = typer.Option(..., "--input", "-i", help="Input file path"),
        domain: str = typer.Option("generic", "--domain", "-d", help="Domain adapter"),
        output: str = typer.Option("output.html", "--output", "-o", help="Output file path"),
        json_output: Optional[str] = typer.Option(
            None, "--json", "-j", help="Export D3 JSON to this path"
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    ):
        """Run the NLP pipeline on input documents."""
        _setup_logging(verbose)

        from graphnlp.pipeline import Pipeline

        typer.echo(f"Processing: {input} (domain={domain})")

        pipe = Pipeline(domain=domain)
        result = pipe.run(input)

        # Output visualization
        result.visualize(output)
        typer.echo(f"Visualization saved to {output}")

        # Optional JSON export
        if json_output:
            result.export_json(json_output)
            typer.echo(f"JSON exported to {json_output}")

        # Print summary
        summary = result.summary()
        typer.echo(f"\n{'─' * 50}")
        typer.echo(f"Nodes: {summary['node_count']}")
        typer.echo(f"Edges: {summary['edge_count']}")
        typer.echo(f"Entities: {summary['entity_count']}")
        typer.echo(f"Triples: {summary['triple_count']}")
        typer.echo(f"Communities: {len(summary.get('communities', []))}")
        typer.echo(f"Avg Sentiment: {summary['avg_sentiment']:.4f}")

        if summary.get("anomalies"):
            typer.echo(f"\nAnomalies ({len(summary['anomalies'])}):")
            for a in summary["anomalies"][:5]:
                typer.echo(f"  • {a['node']} ({a['type']}): {a['sentiment']:.3f}")

    @cli.command()
    def serve(
        host: str = typer.Option("0.0.0.0", help="Host to bind"),
        port: int = typer.Option(8000, help="Port to bind"),
        reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    ):
        """Start the FastAPI API server."""
        import uvicorn

        typer.echo(f"Starting GraphNLP API server on {host}:{port}")
        uvicorn.run(
            "graphnlp.api.app:app",
            host=host,
            port=port,
            reload=reload,
        )

    @cli.command()
    def worker(
        concurrency: int = typer.Option(2, help="Number of worker processes"),
        loglevel: str = typer.Option("info", help="Log level"),
    ):
        """Start a Celery worker for async task processing."""
        from graphnlp.queue.worker import celery_app

        typer.echo(f"Starting Celery worker (concurrency={concurrency})")
        celery_app.worker_main(
            [
                "worker",
                f"--concurrency={concurrency}",
                f"--loglevel={loglevel}",
            ]
        )

    @cli.command("generate-key")
    def generate_key(
        tenant: str = typer.Option(..., "--tenant", "-t", help="Tenant ID"),
    ):
        """Generate an API key for a tenant."""
        import asyncio

        from graphnlp.config import get_settings
        from graphnlp.storage.redis_cache import RedisCache
        from graphnlp.api.auth.api_keys import generate_api_key, set_redis_cache

        async def _generate():
            settings = get_settings()
            cache = RedisCache(settings.redis_url)
            set_redis_cache(cache)
            key = await generate_api_key(tenant)
            await cache.close()
            return key

        key = asyncio.run(_generate())
        typer.echo(f"API key for tenant '{tenant}':")
        typer.echo(f"  {key}")
        typer.echo("\nStore this key securely — it cannot be retrieved again.")

    def main():
        """Entry point for the graphnlp CLI."""
        cli()

    def _setup_logging(verbose: bool = False):
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
