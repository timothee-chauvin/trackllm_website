"""The two papers behind the site, and their BibTeX.

Every page's Cite pill (base.html.j2) and the methodology page read from here,
so a paper is named in one place. Keyed by the method badge the site already
uses ("lt" / "b3it") so a template can pick the paper beside its method.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    authors: tuple[str, ...]
    venue: str
    booktitle: str
    year: int
    arxiv_id: str
    note: str

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def venue_line(self) -> str:
        return f"{self.venue} · arXiv:{self.arxiv_id}"

    @property
    def plain(self) -> str:
        return f'{", ".join(self.authors)} ({self.year}), "{self.title}". {self.venue}. {self.url}'

    @property
    def bibtex(self) -> str:
        fields = {
            "title": self.title,
            "author": " and ".join(self.authors),
            "booktitle": self.booktitle,
            "year": str(self.year),
            "eprint": self.arxiv_id,
            "archivePrefix": "arXiv",
            "url": self.url,
        }
        body = ",\n".join(f"  {k}={{{v}}}" for k, v in fields.items())
        return f"@inproceedings{{{self.key},\n{body}\n}}"


PAPERS: dict[str, Paper] = {
    "lt": Paper(
        key="chauvin2026logprob",
        title="Log Probability Tracking of LLM APIs",
        authors=(
            "Timothée Chauvin",
            "Erwan Le Merrer",
            "François Taïani",
            "Gilles Tredan",
        ),
        venue="ICLR 2026",
        booktitle="International Conference on Learning Representations (ICLR)",
        year=2026,
        arxiv_id="2512.03816",
        note="the LT test, and the TinyChange benchmark",
    ),
    "b3it": Paper(
        key="chauvin2026tokenefficient",
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
        booktitle="International Conference on Machine Learning (ICML)",
        year=2026,
        arxiv_id="2602.11083",
        note="border inputs and B3IT",
    ),
}
