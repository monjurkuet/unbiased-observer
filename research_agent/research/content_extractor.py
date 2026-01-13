from bs4 import BeautifulSoup
import re
import logging
from typing import Optional

logger = logging.getLogger("research_agent.research")


class ContentExtractor:
    """Extract clean text content from various sources"""

    def extract_text(self, content: str, content_type: str = "auto") -> str:
        """Extract text content based on content type"""

        if content_type == "auto":
            content_type = self._detect_content_type(content)

        logger.info(f"Extracting text from {content_type}")

        if content_type == "html":
            return self._extract_from_html(content)
        elif content_type == "markdown":
            return self._extract_from_markdown(content)
        else:
            return self._extract_from_plain_text(content)

    def _detect_content_type(self, content: str) -> str:
        """Auto-detect content type"""

        if content.strip().startswith("<"):
            return "html"
        elif "```" in content or "**" in content:
            return "markdown"
        else:
            return "plain"

    def _extract_from_html(self, html: str) -> str:
        """Extract clean text from HTML"""

        soup = BeautifulSoup(html, "html.parser")

        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()

        text = soup.get_text(separator="\n")

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+\n", "\n", text)

        return text.strip()

    def _extract_from_markdown(self, markdown: str) -> str:
        """Extract text from markdown (simplified)"""

        text = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)

        text = re.sub(r"[*_`#]+", "", text)

        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _extract_from_plain_text(self, text: str) -> str:
        """Extract and clean plain text"""

        text = re.sub(r"\n{3,}", "\n\n", text)

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+\n", "\n", text)

        return text.strip()

    def truncate_to_max_length(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length"""

        if len(text) <= max_length:
            return text

        logger.warning(f"Truncating content from {len(text)} to {max_length} chars")
        return text[:max_length]
