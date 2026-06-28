# RAG-Based Personalized Product Recommendation System

## Overview
This project is an intelligent, hybrid product recommendation engine designed to provide highly personalized user experiences. It combines the reasoning capabilities of Large Language Models (LLMs) with high-speed vector similarity search and Explainable AI (XAI). By analyzing user interaction data (clicks) against a comprehensive product catalog, the system delivers highly accurate recommendations while explaining the logic behind every suggestion.

## Tech Stack
* **Language:** Python
* **Generative AI:** Google Gemini API (`gemini-2.0-flash-exp`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Machine Learning:** Scikit-learn (TF-IDF, MinMaxScaler, RandomForestRegressor)
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
* **Data Processing:** Pandas, NumPy

## Architecture & Workflow

### 1. Advanced Feature Engineering
The system ingests raw product data and user interaction logs from the database. It constructs a unified feature space by utilizing a `ColumnTransformer` to process both text and numeric data simultaneously:
* **Text Features:** Product names, descriptions, categories, and tags are combined and vectorized using `TfidfVectorizer`.
* **Numeric Features:** Pricing structures (`min_price`, `max_price`) are normalized using `MinMaxScaler`.

### 2. Primary Recommendation Engine (LLM)
The core recommendation pipeline utilizes the Google Gemini API. It dynamically constructs prompts containing the user's click history and the available product catalog, leveraging the LLM's contextual understanding to generate a curated list of top 10 relevant product IDs. 

### 3. Vector Search & First-Order Logic Fallback
To ensure high availability and robust performance, the system includes a sophisticated fallback mechanism. If the LLM pipeline encounters an error or returns empty results, the system seamlessly transitions to a vector-based search:
* Converts the product catalog into dense `float32` vectors.
* Stores and indexes the vectors using `faiss.IndexFlatL2` for ultra-fast Euclidean distance calculations.
* Retrieves the closest matching products based on the mathematical similarity to the user's previously clicked items, combined with First-Order Logic (FOL) rules.

### 4. Explainable AI (XAI) Integration
A standout feature of this system is its transparency. It utilizes SHAP to generate natural language explanations for the recommendations. 
* It trains a `RandomForestRegressor` on the user's liked and recommended feature vectors.
* Extracts feature contribution values via the SHAP explainer.
* Generates clear, user-facing explanations detailing exactly *why* an item was recommended (e.g., highlighting the specific textual or pricing features that influenced the model).

## Key Features
* **Hybrid Architecture:** Blends Generative AI with traditional Machine Learning vector search.
* **Fault Tolerance:** Automated fallback to FAISS indexing guarantees recommendations are always served.
* **Transparency:** SHAP integration demystifies the "black box" of AI, explaining recommendations directly to the user.
