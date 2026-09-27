# Security Policy

## Supported code

Security fixes are developed against the current `main` branch. This project
is under active development; older research branches and pre-release chunks
do not have a separate maintenance commitment. Include the exact commit or
release when reporting an issue.

## Report a vulnerability

Use [GitHub's private vulnerability reporting form](https://github.com/AntiO2/vclasp/security/advisories/new).
If that form is unavailable, email [antio2@qq.com](mailto:antio2@qq.com) with
`VClasp security report` in the subject.

Please include:

- The affected revision and platform, including FFmpeg/libavcodec versions.
- A minimal reproduction and the expected versus observed behavior.
- The impact and any relevant logs or stack traces.
- Whether the issue involves chunk parsing, codec execution, object-store
  transport, credentials, or the C/Python interface.

Share malicious or sensitive media through the private report. Remove access
keys, tokens, private endpoints, and personal data from diagnostics. Do not
post an exploit or sensitive input in a public issue while the report is
being investigated.

The maintainer will coordinate investigation and disclosure through the
reporting channel. This project does not currently offer a bug bounty or a
guaranteed response time.

## Operational considerations

VClasp parses binary chunk metadata and passes compressed media to native
codec libraries. Treat untrusted media as untrusted input: use appropriate
process isolation and resource limits, and keep codec dependencies updated.
Use least-privilege object-store credentials and TLS for remote endpoints.
