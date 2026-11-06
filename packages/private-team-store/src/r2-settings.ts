// The R2 settings of the private store are the shared ones
// (`@areneon/object-store`): environment-only credentials, an optional session
// token, `credentials_on_argv` before the environment is read.
export { CREDENTIAL_REFUSALS, CredentialError, type CredentialRefusal, R2Settings, type SignedRequestInit } from "@areneon/object-store";
