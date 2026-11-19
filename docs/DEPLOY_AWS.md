# Deploying to AWS Lambda

The same Docker image used locally and in CI runs on AWS Lambda behind a public
**function URL**. Lambda scales to zero: when nobody uses the demo, nothing runs.
Everything is described in [`template.yaml`](../template.yaml) (AWS SAM):

| Resource | What it is |
|---|---|
| `ApiFunction` | The Docker image on Lambda (arm64, 2 GB memory, 60 s timeout) with a public URL. Its IAM role may call one Bedrock model: Ministral 3 14B in Ireland |
| `ApiLogGroup` | CloudWatch log group for the request logs, kept 30 days |
| `Dashboard` | CloudWatch dashboard built from the request logs (see [Monitoring](#monitoring)) |

How the image works on Lambda (no change to the FastAPI code):
- The [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) in the image turns Lambda events into normal HTTP requests to uvicorn. Outside Lambda it does nothing.
- Lambda's disk is read-only except `/tmp`. Qdrant's local mode writes a lock file, so at start-up the index (~40 MB) is copied to `/tmp` (`index.py`). The model is never downloaded at run time (`HF_HUB_OFFLINE=1`).
- CI runs the image the same way (read-only disk, writable `/tmp` only, unknown user) on both x86_64 and arm64, so these problems show up before a deployment.

## One-time setup (you do this: accounts and credentials are yours)

This was done with a new AWS account from the current sign-up page, the kind AWS calls a
**project** (the project's name is shown at the top right of the console).
Such an account comes ready to use, with two limits that matter here:

- It only works in **the Region AWS assigned it**, chosen from your contact address:
  `eu-north-1` (Stockholm) for Europe, `us-east-2` for the Americas, `ap-southeast-2` for
  Asia-Pacific. Other Regions, such as Paris, are blocked. Check yours in
  [AWS Settings](https://settings.aws.com) → *Project* → *Additional info* → *Region*.
- IAM Identity Center is not available, and you don't need it: the CLI signs in with your
  normal console login (`aws login`).

Lambda, ECR, CloudFormation, S3, IAM and CloudWatch, everything this deployment uses, are
included in the free plan of these accounts
([list](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html)).

1. **Tools on your Mac:**
   ```bash
   brew install awscli aws-sam-cli
   ```
   and [Docker Desktop](https://www.docker.com/products/docker-desktop/) (start it once).
   `aws login` needs AWS CLI 2.32 or later (`aws --version`).
2. **Sign the CLI in.** It opens the browser, where you pick your project. When it asks for a
   Region, type your project's Region (for example `eu-north-1`):
   ```bash
   aws login
   ```
   Check it with `aws sts get-caller-identity`: it prints your account number. The sign-in
   lasts up to 12 hours; after that, run `aws login` again.

No API key is needed: `/ask` uses Ministral 3 14B (by Mistral AI) on **Amazon Bedrock**, and
the function calls it with its own IAM role. A new account can call Bedrock once AWS has
verified it, usually within two hours of sign-up. Bedrock is also included in these accounts'
free plan. The first call to a third-party model accepts its terms of use for the account
([AWS third-party model terms](https://aws.amazon.com/legal/bedrock/third-party-models/)).

With an older ("classic") AWS account, any Region works and you would sign the CLI in with
IAM Identity Center (`aws configure sso`) instead. The rest is the same.

## Deploy

```bash
./scripts/deploy_aws.sh
```

It first checks that the CLI is signed in, that a Region is set and that Docker is running,
and prints the account and Region it will use. Then it runs `sam build` (builds the Docker
image on your Mac, about 5–10 minutes the first time) and `sam deploy` (pushes the image to ECR and creates or updates the stack). SAM shows the
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
image in ECR costs about $0.10 per GB per month. On the free plan of a new AWS account,
nothing is charged unless you upgrade to the paid plan; on the paid plan, set a **spend limit**
in AWS Settings (near the limit, AWS blocks new Lambda calls). With a classic account, create
a *zero spend budget* in Billing → Budgets to get an email as soon as anything costs money.
The model costs $0.24 per million tokens in and out (Ministral 3 14B, Ireland, November 2026);
an `/ask` call of about 3,000 tokens costs under $0.001. The dashboard's cost column
shows this estimate per day.

The URL is public: anyone who has it can call the API. If it is abused, stop all traffic at
once with
```bash
aws lambda put-function-concurrency --function-name <function name> --reserved-concurrent-executions 0
```
(the function name is in the Lambda console), or remove the stack with `sam delete`.

## If something goes wrong

- **"Your account is currently being verified"** from Bedrock: new accounts wait up to about two
  hours after sign-up. `/ask` returns errors until then; `/search` works.
- **"explicit deny in a service control policy"** during the deploy: the CLI points at a Region
  your project can't use. Set the right one (`aws configure set region eu-north-1`) and deploy
  again.
- **"Unable to locate credentials"** or an expired token: run `aws login` again. If the SAM CLI
  still can't read the login, let the AWS CLI hand it the credentials: add a profile with
  `credential_process = aws configure export-credentials --profile default --format process` (and your `region`)
  to `~/.aws/config` and run the script with `AWS_PROFILE` set to it
  ([details](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html#cli-configure-sign-in-cached-credentials)).
- **403 Forbidden on the URL:** the function needs a resource policy allowing both
  `lambda:InvokeFunctionUrl` and `lambda:InvokeFunction`. SAM normally creates it; check
  Lambda console → Configuration → Permissions. Update the SAM CLI if it is old.
- **Timeout on the first call:** a cold start plus an `/ask` call can take several seconds;
  call `/health` first, then retry.
- **Errors in the logs:** CloudWatch → Log groups → `/aws/lambda/fr-en-rag-assistant`.
