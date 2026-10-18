import { Ajv2020 } from "ajv/dist/2020.js";
import type { Profile } from "./profile.ts";

/**
 * The validator instance for a profile (ADR-008 §2). Engine-mirror schemas get
 * their own instance with `allowUnionTypes`; contract schemas never do.
 */
export function ajvFor(profile: Profile): Ajv2020 {
  return new Ajv2020({
    strict: true,
    allErrors: true,
    discriminator: true,
    ...(profile === "engine-mirror" ? { allowUnionTypes: true } : {}),
  });
}
