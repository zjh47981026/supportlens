# Security and limitations

SupportLens is intended for a single user on a trusted computer. The HTTP server binds to 127.0.0.1; do not expose it to a public network or use it as a multi-user service. It has no user authentication or tenant isolation. A local process with access to the user's files can read stored tickets, queries, and reviewed drafts.

POST requests require a per-process token, an allowed Host, and an allowed Origin when present. Static assets use a same-origin content security policy. Imported and model-produced text is displayed as plaintext. Request bodies, ticket fields, model responses, timeouts, batch sizes, and active jobs are bounded. Ollama traffic uses fixed loopback endpoints, disables proxy inheritance, and rejects redirects.

These measures reduce browser-origin abuse; they are not protection against malicious local software. Data is stored unencrypted in the runtime directory. Treat ticket content and model output as untrusted. Source checks validate quotations and source membership, not semantic truth or safety. Approval is a local saved review and does not contact customers.

The source repository and sample corpus contain no customer data or credentials. Runtime data, virtual environments, and .env files are ignored. Only import data you have permission to process. The application provides no automatic deletion or retention interface; use a separate workspace for demos and manage its files deliberately while the app is stopped.

For a public deployment, add authentication, authorization, TLS, rate limits, deliberate data retention, monitoring, tenant isolation, and deployment-specific threat modeling first.
