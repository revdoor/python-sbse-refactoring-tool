# Python SBSE Refactoring Tool

## Overview

A refactoring tool for Python code using search-based software engineering (SBSE).
Uses NSGA-II genetic algorithm to find refactoring sequences that improve code quality across multiple objectives.

## Architecture

| Module | Description |
|--------|-------------|
| Candidate Generator | Detects valid refactoring opportunities via AST analysis |
| Applier | Applies refactoring operators to AST while preserving behavior |
| Metric Calculator | Computes structural and semantic quality metrics |
| NSGA-II Runner | Evolves refactoring sequences via multi-objective optimization |

## Refactoring Operators

- Inline Method (IM)
- Decompose Conditional (DC)
- Reverse Conditional Expression (RC)
- Consolidate Conditional Expression (CC)
- Replace Nested Conditional (RNC)
- Rename Method / Rename Field (RM, RF)
- Extract Method (EM / EMR)
- Remove Duplicate Method (RDM)

## Metrics

- **Structural:** Cyclomatic complexity, SLOC, Fan-in
- **Cost:** Number of refactorings
- **Semantic:** LLM-based readability score (local LLM)

## Requirements

- Python 3.10+
- ollama

## References

Based on methodologies from Ouni et al. (CSMR 2013) and Fakhoury et al. (ICPC 2019).

## ETC

This repository is for a team project in the course "CS454 AI Based Software Engineering" in 2025 Autumn in KAIST.
