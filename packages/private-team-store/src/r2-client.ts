// The R2 client of the private store: the shared client
// (`@areneon/object-store`) bound to this store's layout. Requests, refusals
// and messages are the shared client's.

import { type FetchLike, type R2ClientOptions, R2ObjectStore as SharedR2ObjectStore, type R2Settings } from "@areneon/object-store";
import { PRIVATE_LAYOUT } from "./object-store.ts";

export type { FetchLike, R2ClientOptions };

export class R2ObjectStore extends SharedR2ObjectStore {
  constructor(settings: R2Settings, options: R2ClientOptions = {}) {
    super(settings, PRIVATE_LAYOUT, options);
  }
}
