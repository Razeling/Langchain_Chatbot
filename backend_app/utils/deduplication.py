"""
Centralized deduplication utilities for the Car Troubleshooting Chatbot API.
"""


def deduplicate_sources(sources):
    """Remove duplicate sources based on content similarity."""
    if not sources:
        return []

    unique_sources = []
    seen_content = set()

    for source in sources:
        # Create a simplified version for comparison
        if hasattr(source, "content"):
            content_key = source.content.lower().strip()[:100]  # First 100 chars
        elif hasattr(source, "page_content"):
            content_key = source.page_content.lower().strip()[:100]
        else:
            content_key = str(source).lower().strip()[:100]

        if content_key not in seen_content:
            seen_content.add(content_key)
            unique_sources.append(source)

    return unique_sources
