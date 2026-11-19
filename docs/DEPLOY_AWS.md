# Deploying to AWS Lambda

The same Docker image used locally and in CI runs on AWS Lambda behind a public
**function URL**. Lambda scales to zero: when nobody uses the demo, nothing runs.
Everything is described in [`template.yaml`](../template.yaml) (AWS SAM):

| Resource | What it is |
|---|---|
| `ApiFunction` | The Docker image on Lambda (arm64, 2 GB memory, 60 s timeout) with a public URL |
| `ApiLogGroup` | CloudWatch log group for the request logs, kept 30 days |
| `Dashboard` | CloudWatch dashboard built from the request logs (see [Monitoring](#monitoring)) |

How the image works on Lambda (no change to the FastAPI code):
- The [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) in the image turns Lambda events into normal HTTP requests to uvicorn. Outside Lambda it does nothing.
- Lambda's disk is read-only except `/tmp`. Qdrant's local mode writes a lock file, so at start-up the index (~40 MB) is copied to `/tmp` (`index.py`). The model is never downloaded at run time (`HF_HUB_OFFLINE=1`).
- CI runs the image the same way (read-only disk, writable `/tmp` only, unknown user) on both x86_64 and arm64, so these problems show up before a deployment.

## One-time setup (you do this: accounts and credentials are yours)

1. **AWS account.** Create it at [aws.amazon.com/free](https://aws.amazon.com/free). It asks for a card.
   - Turn on MFA for the root user, then use the root user only for billing.
   - Billing → Budgets → *Create budget* → **Zero spend budget** (an email as soon as anything costs money).
2. **A daily-use login.** In *IAM Identity Center*, enable it, create a user for yourself with the
   `AdministratorAccess` permission set on your account. Note the **AWS access portal URL**.
3. **Tools on your Mac:**
   ```bash
   brew install awscli aws-sam-cli
   ```
   and [Docker Desktop](https://www.docker.com/products/docker-desktop/) (start it once).
4. **Sign the CLI in** (opens the browser; choose region `eu-west-3`, profile name `default`):
   ```bash
   aws configure sso
   ```
   Later sessions only need `aws sso login`.
5. **Mistral key** in the local `.env` file (`cp .env.example .env`, then paste the key). Optional:
   without it, the deployed API serves `/search` only and `/ask` returns 503.

## Deploy

```bash
./scripts/deploy_aws.sh
```

It runs `sam build` (builds the Docker image on your Mac, about 5–10 minutes the first time)
and `sam deploy` (pushes the image to ECR and creates or updates the stack). SAM shows the
list of changes and asks for confirmation. At the end it prints:

- `ApiUrl`: the public URL. Open it in a browser for the demo page.
- `DashboardUrl`: the CloudWatch dashboard.

Check it:

```bash
./scripts/smoke_test.sh <ApiUrl>
```

The first request after a quiet period is slower (a *cold start*: a new Lambda instance
loads the image, the index and the embedding model). The dashboard counts cold starts.

To update after a code change, run `./scripts/deploy_aws.sh` again.
To remove everything: `sam delete`.

## Monitoring

Every `/ask` and `/search` call writes one JSON line (see `src/rag_assistant/monitoring.py`).
Lambda sends these lines to CloudWatch Logs, and the dashboard queries them with
CloudWatch Logs Insights:

- requests per hour by endpoint
- latency (median and 95th percentile) by endpoint and language
- failures (status 500+) by error type
- `/ask` answers with and without a citation, by language
- tokens and estimated cost per day
- retrieval score by method
- cold starts, and Lambda's own invocations / errors / throttles
- the last 20 requests

The question and the answer are never logged, only their length.

To read the raw logs from the terminal: `sam logs --stack-name fr-en-rag-assistant --tail`.

## Cost

At demo traffic this should stay within AWS's always-free allowances (Lambda: 1 million
requests and 400,000 GB-seconds per month; CloudWatch: 5 GB of logs and 3 dashboards). The
image in ECR costs about $0.10 per GB per month. The zero-spend budget emails you if anything
is charged. Mistral's free plan costs nothing; the dashboard's cost column shows what the
same traffic would cost on a paid plan.

The URL is public: anyone who has it can call the API. If it is abused, stop all traffic at
once with
```bash
aws lambda put-function-concurrency --function-name <function name> --reserved-concurrent-executions 0
```
(the function name is in the Lambda console), or remove the stack with `sam delete`.

## If something goes wrong

- **403 Forbidden on the URL:** the function needs a resource policy allowing both
  `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction`. SAM normally creates it; check
  Lambda console → Configuration → Permissions. Update the SAM CLI if it is old.
- **Timeout on the first call:** a cold start plus an `/ask` call can take several seconds;
  call `/health` first, then retry.
- **Errors in the logs:** CloudWatch → Log groups → `/aws/lambda/fr-en-rag-assistant`.
