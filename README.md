# Titanic Analytics Capstone Project

This repository contains a full end‑to‑end data science and analytics project built around the classic Titanic dataset, from data cleaning and feature engineering through modeling, dashboarding, and an LLM-powered assistant.

## Project Overview

The goal of this project is to explore passenger survival on the Titanic and demonstrate a modern analytics workflow: preparing raw data, training predictive models, building an interactive dashboard for insights, and exposing the project via a language model interface.[page:1] The code is organized into four main parts plus a setup guide to help you get started quickly.

## Repository Structure

- `part1-data-cleaning`: Notebooks and scripts for data ingestion, cleaning, exploration, and feature engineering on the Titanic dataset.[page:1]
- `part2-model`: Model training, evaluation, and experimentation (e.g., classification models to predict passenger survival).[page:1]
- `part3-dashboard`: Code for building an interactive dashboard that visualizes key features, survival patterns, and model outputs.[page:1]
- `part4-llm`: An LLM-based app that lets users ask natural-language questions about the data, analysis, and results.
- `SETUP_GUIDE.md`: A detailed setup and environment guide for running each project component locally.

## Key Features

- End-to-end workflow from raw data to deployable analytics artifacts.
- Modular structure (Parts 1–4) so you can focus on data prep, modeling, visualization, or LLM integration independently.
- Reproducible environments defined via `requirements.txt` files in each part.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/PVF88/titanic-analytics.git
   cd titanic-analytics
   ```
2. Read `SETUP_GUIDE.md` for environment setup, installation instructions, and how to run each part of the project.
3. Start with `part1-data-cleaning`, then proceed through `part2-model`, `part3-dashboard`, and `part4-llm` as needed.

## Technologies Used

- Python (primary language for all components).
- Common data science libraries (e.g., pandas, NumPy, scikit-learn, plotting libraries) as specified per part in `requirements.txt`.
- LLM tooling/frameworks for the `part4-llm` app, as described in that folder’s documentation.

## Intended Audience

This project is designed for students and practitioners who want a concrete, hands-on example of a capstone-level analytics solution using a well-known dataset.[page:1] It can be used for learning, portfolio building, or as a template for similar end-to-end data science projects.

## License

This project is currently private on GitHub; please check the repository settings or contact the owner for licensing and reuse details.
