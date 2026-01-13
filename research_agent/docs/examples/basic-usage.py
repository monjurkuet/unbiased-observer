#!/usr/bin/env python3
"""
Basic Usage Examples for the Autonomous Research Agent

This script demonstrates common usage patterns for interacting with
the research agent programmatically.
"""

import asyncio
import os
from typing import List, Dict, Any

# Set PYTHONPATH for imports
import sys
sys.path.insert(0, '/home/administrator/dev/unbiased-observer')

from research_agent.config import load_config
from research_agent.monitoring import setup_logging
from research_agent.research import ManualSourceManager
from research_agent.orchestrator import TaskQueue
from research_agent.monitoring import MetricsCollector
from research_agent.ingestion import IngestionPipeline


async def example_1_add_research_content():
    """
    Example 1: Adding research content manually

    This shows how to add different types of research content
    to the agent's processing queue.
    """
    print("=== Example 1: Adding Research Content ===")

    # Initialize components
    config = load_config()
    logger, _, _, _ = setup_logging(config)

    queue = TaskQueue(config, logger)
    await queue.initialize()

    manager = ManualSourceManager(queue)

    # Add different types of content
    examples = [
        {
            "type": "url",
            "url": "https://arxiv.org/pdf/2301.07041.pdf",
            "metadata": {"title": "Attention Is All You Need", "year": 2023}
        },
        {
            "type": "text",
            "text": """
            Recent advances in transformer architectures have revolutionized
            natural language processing. The key innovation is the multi-head
            attention mechanism that allows the model to attend to different
            parts of the input simultaneously.
            """,
            "metadata": {"topic": "transformers", "source": "notes"}
        }
    ]

    for example in examples:
        if example["type"] == "url":
            task_id = await manager.add_url_source(
                url=example["url"],
                metadata=example["metadata"]
            )
            print(f"Added URL source: {example['url']} (Task ID: {task_id})")
        elif example["type"] == "text":
            task_id = await manager.add_text_source(
                text=example["text"],
                metadata=example["metadata"]
            )
            print(f"Added text content (Task ID: {task_id})")

    await queue.close()
    print("Content addition complete!\n")


async def example_2_monitor_system_status():
    """
    Example 2: Monitoring system status and metrics

    This demonstrates how to check the agent's current status
    and performance metrics.
    """
    print("=== Example 2: System Monitoring ===")

    config = load_config()
    metrics = MetricsCollector(config)

    # Get comprehensive metrics
    summary = await metrics.get_summary_metrics()

    print("System Status:")
    print(f"  Database Connected: {summary['system_health']['database_connected']}")
    print(f"  Total Entities: {summary['ingestion_metrics']['entities_extracted']:,}")
    print(f"  Total Relationships: {summary['ingestion_metrics']['relationships_extracted']:,}")
    print(f"  Communities Found: {summary['processing_metrics']['total_communities']}")

    # Task queue status
    task_metrics = summary['task_metrics']
    print("
Task Queue:")
    print(f"  Total Tasks: {task_metrics['total_tasks']}")
    print(f"  Pending: {task_metrics['pending_tasks']}")
    print(f"  Completed: {task_metrics['completed_tasks']}")
    print(f"  Failed: {task_metrics['failed_tasks']}")
    print(".1f")

    print("System monitoring complete!\n")


async def example_3_query_knowledge_graph():
    """
    Example 3: Querying the knowledge graph

    This shows how to query the knowledge graph directly
    using SQL queries.
    """
    print("=== Example 3: Knowledge Graph Queries ===")

    import psycopg
    from psycopg.rows import dict_row

    config = load_config()
    conn_str = config.database.connection_string

    async with await psycopg.AsyncConnection.connect(conn_str) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Query 1: Recent entities
            await cur.execute("""
                SELECT id, name, type, created_at
                FROM nodes
                ORDER BY created_at DESC
                LIMIT 5
            """)

            print("Recent Entities:")
            async for row in cur:
                print(f"  {row['name']} ({row['type']}) - {row['created_at']}")

            # Query 2: Popular concepts
            await cur.execute("""
                SELECT n.name, COUNT(e.id) as connections
                FROM nodes n
                LEFT JOIN edges e ON n.id = e.source_id OR n.id = e.target_id
                WHERE n.type = 'concept'
                GROUP BY n.id, n.name
                ORDER BY connections DESC
                LIMIT 5
            """)

            print("
Popular Concepts:")
            async for row in cur:
                print(f"  {row['name']}: {row['connections']} connections")

            # Query 3: Research communities
            await cur.execute("""
                SELECT name, size, summary
                FROM communities
                ORDER BY size DESC
                LIMIT 3
            """)

            print("
Research Communities:")
            async for row in cur:
                summary_preview = row['summary'][:100] + "..." if row['summary'] and len(row['summary']) > 100 else row['summary']
                print(f"  {row['name']} ({row['size']} members)")
                if summary_preview:
                    print(f"    {summary_preview}")

    print("Knowledge graph queries complete!\n")


async def example_4_batch_processing():
    """
    Example 4: Batch processing of multiple documents

    This demonstrates processing multiple research documents
    in a batch operation.
    """
    print("=== Example 4: Batch Processing ===")

    config = load_config()
    logger, _, _, _ = setup_logging(config)

    # Initialize ingestion pipeline
    pipeline = IngestionPipeline(config, logger)
    await pipeline.initialize()

    # Sample research documents
    documents = [
        {
            "content": """
            Convolutional Neural Networks (CNNs) have revolutionized computer vision.
            The key innovation is the use of convolutional filters that can detect
            local patterns in images. These networks are particularly effective
            for image classification, object detection, and segmentation tasks.
            """,
            "metadata": {"topic": "computer_vision", "source": "cnn_overview"}
        },
        {
            "content": """
            Reinforcement Learning (RL) algorithms learn through interaction
            with an environment. The agent takes actions and receives rewards,
            learning optimal policies through trial and error. Deep RL combines
            neural networks with RL, enabling complex decision-making.
            """,
            "metadata": {"topic": "reinforcement_learning", "source": "rl_overview"}
        }
    ]

    total_entities = 0
    total_relationships = 0

    for i, doc in enumerate(documents, 1):
        print(f"Processing document {i}/{len(documents)}...")

        result = await pipeline.ingest_content(
            content=doc["content"],
            metadata=doc["metadata"]
        )

        print(f"  Entities: {result['entities_stored']}")
        print(f"  Relationships: {result['relationships_stored']}")
        print(".2f")

        total_entities += result['entities_stored']
        total_relationships += result['relationships_stored']

    print("
Batch processing summary:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Total entities: {total_entities}")
    print(f"  Total relationships: {total_relationships}")

    await pipeline.close()
    print("Batch processing complete!\n")


async def example_5_custom_arxiv_search():
    """
    Example 5: Custom arXiv search and monitoring

    This shows how to perform custom searches on arXiv
    and add results to the processing queue.
    """
    print("=== Example 5: Custom arXiv Search ===")

    from research_agent.research import ArxivIntegrator

    config = load_config()
    logger, _, _, _ = setup_logging(config)

    integrator = ArxivIntegrator(config, logger)

    # Perform custom search
    search_configs = [
        {
            "name": "Recent AI Papers",
            "keywords": ["artificial intelligence", "machine learning"],
            "max_results": 3,
            "days_back": 7
        },
        {
            "name": "NLP Research",
            "keywords": ["natural language processing", "transformers"],
            "max_results": 2,
            "days_back": 3
        }
    ]

    total_papers = 0

    for search_config in search_configs:
        print(f"Searching for: {search_config['name']}")

        papers = await integrator.search_by_keywords(
            keywords=search_config["keywords"],
            max_results=search_config["max_results"],
            days_back=search_config["days_back"]
        )

        print(f"  Found {len(papers)} papers")

        for paper in papers:
            print(f"    - {paper['title']}")
            print(f"      Authors: {', '.join(paper['authors'])}")
            print(f"      URL: {paper['pdf_url']}")

        total_papers += len(papers)

    print(f"\nTotal papers discovered: {total_papers}")

    # Optionally add to processing queue
    if total_papers > 0:
        queue = TaskQueue(config, logger)
        await queue.initialize()

        for search_config in search_configs:
            papers = await integrator.search_by_keywords(
                keywords=search_config["keywords"],
                max_results=search_config["max_results"],
                days_back=search_config["days_back"]
            )

            for paper in papers:
                task_id = await integrator.add_paper_to_queue(paper, queue)
                print(f"Queued paper: {paper['title']} (Task ID: {task_id})")

        await queue.close()

    print("arXiv search complete!\n")


async def main():
    """
    Run all examples in sequence.
    """
    print("Autonomous Research Agent - Basic Usage Examples")
    print("=" * 50)
    print()

    try:
        await example_1_add_research_content()
        await example_2_monitor_system_status()
        await example_3_query_knowledge_graph()
        await example_4_batch_processing()
        await example_5_custom_arxiv_search()

        print("All examples completed successfully!")
        print("\nNext steps:")
        print("- Start the web UI: cd ui && python run.py")
        print("- View logs: tail -f logs/agent.log")
        print("- Monitor tasks: watch -n 5 'psql -d knowledge_graph -c \"SELECT status, COUNT(*) FROM research_tasks GROUP BY status;\"'")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if running in correct environment
    if not os.path.exists('/home/administrator/dev/unbiased-observer'):
        print("Error: This script must be run from the correct environment.")
        print("Expected path: /home/administrator/dev/unbiased-observer")
        sys.exit(1)

    # Run examples
    asyncio.run(main())