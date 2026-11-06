// The in-memory object store of the private store's tests: the shared store
// (`@areneon/object-store`) bound to this store's layout.

import { InMemoryObjectStore as SharedInMemoryObjectStore } from "@areneon/object-store";
import { PRIVATE_LAYOUT } from "./object-store.ts";

export class InMemoryObjectStore extends SharedInMemoryObjectStore {
  constructor() {
    super(PRIVATE_LAYOUT);
  }
}
