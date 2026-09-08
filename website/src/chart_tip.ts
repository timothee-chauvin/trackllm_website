// The floating readout both charts share: the endpoint page's per-lane one and the
// model page's shared timeline. Same box, same placement rule, same wording of a
// date and a value -- a reader who learned one reads the other without being told.
import { esc } from "./components";

export const MARK_R = 3.4;
const TIP_DX = 14; // the tip sits beside the pointer, never under it
const TIP_DY = -10;

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

/** Beside the pointer and inside `wrap`, the positioned box the tip is absolute in. */
export function placeTip(tip: HTMLElement, wrap: HTMLElement, ev: PointerEvent): void {
  const box = wrap.getBoundingClientRect();
  const left = ev.clientX - box.left + TIP_DX;
  tip.style.left = `${Math.max(0, Math.min(left, box.width - tip.offsetWidth))}px`;
  tip.style.top = `${ev.clientY - box.top + TIP_DY}px`;
}
