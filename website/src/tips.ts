// Entry point for pages with no other script: their badges are server-rendered
// (headline_badge, _macros.html.j2) but still need bindTips wired up once. Pages
// that already load an entry point (overview.js, model.js, provider.js, the
// tracked branch of endpoint.js) bind it themselves instead -- see components.ts::
// bindTips for why one binding per page is what the shared popover wants.
import { bindTips } from "./components";

bindTips(document.body);
