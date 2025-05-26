# Langchain B12

This repo hosts a collection of custom LangChain components.

## Installation

To install this package, run
```bash
pip install langchain-b12
```

Some components rely on additional packages that may be installed as extras.
For example, to use the Google chatmodel `ChatGenAI`, you can run
```bash
pip install langchain-b12[google]
```

## Components

The repo contains these components:

- `ChatGenAI`: An implementation of `BaseChatModel` that uses the `google-genai` package. Note that `langchain-google-genai` and `langchain-google-vertexai` exist, but neither uses the latest and recommended `google-genai` package.
- `GenAIEmbeddings`: An implementation of `Embeddings` that uses the `google-genai` package.

## Comments

This repo exists for easy reuse and extension of custom LangChain components.
When appropriate, we will create PRs to merge these components directly into LangChain.
