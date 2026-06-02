# Semantic Kernel agents (`sk/`)

Sample [Semantic Kernel](https://github.com/microsoft/semantic-kernel) agents that call **Azure OpenAI** (Azure AI Foundry) using credentials from a `.env` file.

## Prerequisites

- Python 3.10+
- Virtual environment at the repo root with dependencies installed (`semantic-kernel`, `python-dotenv`, `openai`)
- Azure OpenAI deployment and API key

## Configuration

Create or edit `sk/.env` (or run from the repo root with a root `.env`):

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Foundry URL, often `https://<resource>.services.ai.azure.com/openai/v1/` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment / model name (e.g. `gpt-5-nano`) |
| `AZURE_OPENAI_API_KEY` | API key for the deployment |
| `AZURE_OPENAI_API_VERSION` | Optional. Used by Azure connector scripts. Default: `2024-10-21` |

Do not commit real API keys to source control.

### Endpoint formats

| Connector | How the scripts use your URL |
|-----------|-------------------------------|
| `OpenAIChatCompletion` | Uses the URL as-is (OpenAI-compatible `/openai/v1` base URL). |
| `AzureChatCompletion` | Strips `/openai/v1` (and similar suffixes) so the **resource root** is used as `endpoint` (e.g. `https://<resource>.services.ai.azure.com`). |

## Scripts

| File | Description |
|------|-------------|
| `sk_agent_openai.py` | Minimal agent using `OpenAIChatCompletion` + `AsyncOpenAI` against the Foundry `/openai/v1` endpoint. Single prompt, prints one reply. |
| `sk_agent_azure.py` | Same flow using `AzureChatCompletion` (classic Azure deployment API). Single prompt, prints one reply. |
| `sk_agent_azure_plugin.py` | Interactive chat with a `MenuPlugin` (specials, prices) and optional structured output (`MenuItem`). Type `exit` or `quit` to stop. |
| `sk_agent_azure_multi.py` | Multi-agent demo: `TriageAgent` delegates to `BillingAgent` and `RefundAgent`. Interactive loop; type `exit` to stop. |

## Run

From the `sk` directory (so `load_dotenv()` picks up `sk/.env`):

```powershell
cd sk
..\.venv\Scripts\Activate.ps1

python sk_agent_openai.py
python sk_agent_azure.py
python sk_agent_azure_plugin.py
python sk_agent_azure_multi.py
```

From the repo root (uses root `.env` if present):

```powershell
python sk\sk_agent_azure.py
```

## Which script should I use?

- **OpenAI-compatible Foundry URL** → start with `sk_agent_openai.py`.
- **Native Azure SK connector** → use `sk_agent_azure.py` or the plugin/multi variants.
- **Function calling / plugins** → `sk_agent_azure_plugin.py`.
- **Multiple specialized agents** → `sk_agent_azure_multi.py`.

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `chat_deployment_name is required` | Missing `AZURE_OPENAI_DEPLOYMENT_NAME` in `.env`. |
| `404 Resource not found` with `AzureChatCompletion` | Endpoint still includes `/openai/v1`; use `sk_agent_azure.py` normalization or set resource root + `AZURE_OPENAI_API_VERSION=2024-10-21`. |
| `404` with `OpenAIChatCompletion` | Wrong `AZURE_OPENAI_ENDPOINT` or deployment name. |
| Env vars not loaded | Run from `sk/` or pass an explicit path to `load_dotenv()`. |
