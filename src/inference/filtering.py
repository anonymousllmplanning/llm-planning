#!/usr/bin/env python3
"""
Tool Filtering Utilities for Delta MCP Dataset

Delta datasets can have large tool catalogs (e.g., 2397 tools in old_delta).
This module provides keyword-based filtering to select relevant tools
for a given query, reducing context length and improving model focus.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Set


def extract_keywords_from_query(query: str) -> Set[str]:
    """
    Extract relevant keywords from user query for tool matching.
    """
    # Basic stopwords
    stopwords = {
        'i', 'me', 'my', 'the', 'a', 'an', 'to', 'from', 'in', 'on', 'for',
        'with', 'and', 'or', 'is', 'are', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'can', 'need', 'want', 'please', 'first', 'then', 'after',
        'before', 'if', 'when', 'how', 'what', 'which', 'this', 'that', 'these',
        'those', 'it', 'its', 'of', 'at', 'by', 'as', 'so', 'but', 'not', 'no',
        'yes', 'all', 'any', 'some', 'such', 'into', 'over', 'under', 'above',
        'below', 'between', 'through', 'during', 'until', 'while', 'also', 'just',
        'only', 'own', 'same', 'too', 'very', 'can', 'make', 'check', 'get', 'use'
    }

    # Tokenize and clean
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*\b', query.lower())
    keywords = {w for w in words if w not in stopwords and len(w) > 2}

    # Also extract potential tool-related terms (CamelCase, snake_case)
    camel_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', query)
    keywords.update(t.lower() for t in camel_terms)

    return keywords


def score_tool_relevance(tool: Dict[str, Any], keywords: Set[str]) -> float:
    """
    Score a tool's relevance to the query keywords.
    Higher score = more relevant.
    """
    score = 0.0

    tool_id = (tool.get("tool_id") or "").lower()
    tool_name = (tool.get("tool_name") or "").lower()
    server_name = (tool.get("server_name") or "").lower()
    description = (tool.get("description") or "").lower()

    # Extract words from tool info
    tool_words = set()
    for text in [tool_id, tool_name, server_name, description]:
        tool_words.update(re.findall(r'\b[a-z][a-z0-9_-]*\b', text))

    # Score based on keyword matches
    for kw in keywords:
        if kw in tool_id or kw in tool_name:
            score += 3.0  # Strong match
        elif kw in server_name:
            score += 2.0  # Server match
        elif kw in description:
            score += 1.0  # Description match
        elif any(kw in tw for tw in tool_words):
            score += 0.5  # Partial match

    return score


def filter_relevant_tools(
    tools: List[Dict[str, Any]],
    query: str,
    max_tools: int = 50,
    min_score: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filter and rank tools by relevance to the query.
    Returns at most max_tools tools.

    Args:
        tools: List of tool definitions
        query: User query text
        max_tools: Maximum number of tools to return
        min_score: Minimum relevance score threshold

    Returns:
        Filtered and ranked list of tools
    """
    if len(tools) <= max_tools:
        return tools

    keywords = extract_keywords_from_query(query)

    # Score all tools
    scored_tools = []
    for tool in tools:
        score = score_tool_relevance(tool, keywords)
        if score >= min_score:
            scored_tools.append((score, tool))

    # Sort by score (descending)
    scored_tools.sort(key=lambda x: x[0], reverse=True)

    # Return top tools
    return [t for _, t in scored_tools[:max_tools]]
