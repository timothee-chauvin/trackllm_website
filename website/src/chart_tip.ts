// The floating readout both charts share: the endpoint page's per-lane one and the
// model page's shared timeline. Same box, same placement rule, same wording of a
// date and a value -- a reader who learned one reads the other without being told.
import { esc } from "./components";

export const MARK_R = 3.4;
const TIP_GAP = 1; // between the pointer and the tip's corner: a horizontal sweep never enters it
const VIEW_PAD = 4;

/** One value in the readout, in the colour of the trace it was read off. */
export interface Cell {
  text: string;
  col: string;
}

export function tipHTML(date: string, cells: Cell[]): string {
  return (
    `<span class="d">${esc(date)}</span>` +
    cells.map((c) => `<span class="v" style="color:${c.col}">${esc(c.text)}</span>`).join("")
  );
}

/** The pointer at the tip's top-right corner: the tip hangs below and to the left,
 *  so a sweep along the curve never crosses it and a hand moving down into it (to
 *  copy a line) meets it at once. Flipped to the other side where that side has no
 *  room, and kept inside the viewport rather than `wrap`: a detail block is read
 *  where it opens, never by scrolling the page or the block. */
export function placeTip(tip: HTMLElement, wrap: HTMLElement, ev: PointerEvent): void {
  const box = wrap.getBoundingClientRect();
  const w = tip.offsetWidth;
  const h = tip.offsetHeight;
  const px = ev.clientX - box.left;
  const py = ev.clientY - box.top;
  const viewTop = -box.top + VIEW_PAD;
  const viewBottom = window.innerHeight - box.top - VIEW_PAD;
  let left = px - TIP_GAP - w;
  if (left < 0) left = Math.min(px + TIP_GAP, box.width - w);
  let top = py + TIP_GAP;
  if (top + h > viewBottom) top = py - TIP_GAP - h;
  tip.style.left = `${Math.max(0, left)}px`;
  tip.style.top = `${Math.max(viewTop, top)}px`;
}
