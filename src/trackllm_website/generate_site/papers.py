"""The two papers behind the site, and their citations.

Every page's Cite dialog (base.html.j2) and the methodology page read from
here, so a paper is named in one place. Keyed by the method badge the site
already uses ("lt" / "b3it") so a template can pick the paper beside its
method. The BibTeX is OpenReview's, verbatim.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    title: str
    authors: tuple[str, ...]
    venue: str
    year: int
    arxiv_id: str
    note: str
    bibtex: str

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def venue_line(self) -> str:
        return f"{self.venue} · arXiv:{self.arxiv_id}"

    @property
    def plain(self) -> str:
        return f'{", ".join(self.authors)} ({self.year}), "{self.title}". {self.venue}. {self.url}'


PAPERS: dict[str, Paper] = {
    "lt": Paper(
        title="Log Probability Tracking of LLM APIs",
        authors=(
            "Timothée Chauvin",
            "Erwan Le Merrer",
            "François Taïani",
            "Gilles Tredan",
        ),
        venue="ICLR 2026",
        year=2026,
        arxiv_id="2512.03816",
        note="the LT test, and the TinyChange benchmark",
        bibtex="""@inproceedings{
chauvin2026log,
title={Log Probability Tracking of {LLM} {API}s},
author={Timothee Chauvin and Erwan Le Merrer and Francois Taiani and Gilles Tredan},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=hFxivbAgVP}
}""",
    ),
    "b3it": Paper(
        title="Token-Efficient Change Detection in LLM APIs",
        authors=(
            "Timothée Chauvin",
            "Clément Lalanne",
            "Erwan Le Merrer",
            "Jean-Michel Loubes",
            "François Taïani",
            "Gilles Tredan",
        ),
        venue="ICML 2026",
        year=2026,
        arxiv_id="2602.11083",
        note="border inputs and B3IT",
        bibtex="""@inproceedings{
chauvin2026tokenefficient,
title={Token-Efficient Change Detection in {LLM} {API}s},
author={Timothee Chauvin and Cl{\\'e}ment Lalanne and Erwan Le Merrer and Jean-Michel Loubes and Francois Taiani and Gilles Tredan},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=7cMlZZYZT0}
}""",
    ),
}
