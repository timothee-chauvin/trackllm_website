// The "Copy for LLM" pill (base.html.j2): fetches the page's markdown twin and
// puts it on the clipboard. Loaded on every page, so it binds by class.

const DONE = "Copied";
const FAIL = "Copy failed";
const RESET_MS = 1500;

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

export function bindCopyPills(root: ParentNode): void {
  for (const btn of root.querySelectorAll<HTMLButtonElement>("button.copy-pill")) {
    // The flash labels are shorter than the resting one; without a locked width
    // the pill shrinks on click and everything after it reflows.
    btn.style.width = `${btn.offsetWidth}px`;
    const label = btn.querySelector<HTMLElement>(".copy-pill-label")!;
    const original = label.textContent;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const flash = (text: string, error: boolean): void => {
      label.textContent = text;
      btn.classList.toggle("is-error", error);
      clearTimeout(timer);
      timer = setTimeout(() => {
        label.textContent = original;
        btn.classList.remove("is-error");
      }, RESET_MS);
    };
    btn.addEventListener("click", async () => {
      try {
        let text = btn.dataset.copyText;
        if (text === undefined) {
          const res = await fetch(btn.dataset.src!);
          if (!res.ok) throw new Error(`${btn.dataset.src}: HTTP ${res.status}`);
          text = await res.text();
        }
        await copyText(text);
        flash(DONE, false);
      } catch (err) {
        console.error("copy-llm:", err);
        flash(FAIL, true);
      }
    });
  }
}

bindCopyPills(document);
