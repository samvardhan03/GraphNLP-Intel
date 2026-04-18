#!/usr/bin/env bash
celery -A graphnlp.queue.worker worker --loglevel=info --concurrency=4
