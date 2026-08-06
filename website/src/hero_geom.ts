// Geometry of the Overview hero: the trace's own box, and how wide its full-bleed layers
// have to be on a wide page. Everything here is pure -- no DOM, no measurement beyond the
// numbers overview.ts hands in.
import { DAY_MS } from "./components";

// The trace, in the SVG's own box. Both hero layers are sized to the clear space above the
// stat cards, so the curve is never half-hidden behind them however the hero reflows: zero
// drift sits on the cards' top edge and the fill between them is the area under the curve.
export const HERO_W = 1200;
export const HERO_VB_H = 200; // also the zero line
export const HERO_TOP = 14;
export const HERO_HIT_WIDTH = 18; // invisible fat stroke: the curve is 1.6 units thin
export const HERO_TIP_DX = 16;
export const HERO_TIP_DY = 18;

// On a wide page the changepoint rise lands in the middle of the lede. Stretching the
// full-bleed layers past the right edge slides the rise clear of the text -- the baseline
// then runs under the lede and the rise beside it, framing it instead of crossing it --
// and the hero clips the flat tail that falls off the right edge, which is what pays for
// the room. Below HERO_WIDE there is no such room, so the layers stay at the hero's width.
export const HERO_WIDE = "(min-width: 1440px)";
export const HERO_CLEAR_GAP = 48; // between the lede's right edge and the changepoint rule
const HERO_MAX_STRETCH = 1.25; // past this, too much of the curve's tail would be hidden

/** Layer width, relative to the hero's, that puts the changepoint rise at `clearTo` px. */
export function heroStretch(changeFrac: number, heroWidth: number, clearTo: number): number {
  if (!(changeFrac > 0) || heroWidth <= 0) return 1;
  const stretch = clearTo / changeFrac / heroWidth;
  return stretch > 1 && stretch <= HERO_MAX_STRETCH ? stretch : 1;
}

/** Last day of the window still on screen: what a stretched layer runs off the right edge
 *  is not drawn, and the hover card may not name days the page does not show. */
export function heroDrawnTo(start: string, end: string, stretch: number): string {
  if (stretch <= 1) return end;
  const days = (Date.parse(end) - Date.parse(start)) / DAY_MS;
  return new Date(Date.parse(start) + Math.round(days / stretch) * DAY_MS).toISOString().slice(0, 10);
}
