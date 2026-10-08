# Bedrock-to-OpenAI Proxy

Drop-in OpenAI-compatible API backed by AWS Bedrock. Teams issue their own API keys and point any AI-enabled tool, IDE extension, or script at a shared gateway - without exposing AWS credentials or losing control over which models are used.

## How it works

Exposes `/v1/chat/completions` and `/v1/models` in the OpenAI format. Requests are authenticated with API keys, translated to Bedrock Converse API calls, and responses are translated back. Streaming is supported.

