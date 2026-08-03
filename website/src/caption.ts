/** The drift chart's "Reading it" caption. Most endpoints are monitored by one method
 *  only, so the caption must never point at a lane or a unit the page isn't showing. */
export function readingCaption(hasLT: boolean, hasB3IT: boolean): string {
  const both = hasLT && hasB3IT;
  const lead = both ? "both lanes share the time axis. A" : "a";
  const labeled = both
    ? "the change-detection metric reached (LT: σ) or peak reached (B3IT: TV)"
    : hasLT
      ? "the change-detection metric σ"
      : "the peak total variation (TV) reached";
  return `<b>Reading it:</b> ${lead} change reads as a step up that persists; dashed lines mark detected changepoints, with ${labeled} labeled above.`;
}
