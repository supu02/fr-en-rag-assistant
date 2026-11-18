from pathlib import Path

from rag_assistant.ingest import parse_page, split_lines

PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F999" type="Fiche d'information"
  spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F999">
  <dc:title>Dépôt de garantie</dc:title>
  <dc:description>Une somme versée au propriétaire.</dc:description>
  <dc:date>modified 2026-11-01</dc:date>
  <SousThemePere ID="N1">Bail d’habitation</SousThemePere>
  <ListeSituations>
    <Situation>
      <Titre>Logement vide</Titre>
      <Texte>
        <Chapitre>
          <Titre><Paragraphe>Quel est le montant ?</Paragraphe></Titre>
          <Paragraphe>Au maximum <MiseEnEvidence>1 mois</MiseEnEvidence> de loyer.</Paragraphe>
          <BlocCas>
            <Cas><Titre>Loyer payé chaque trimestre</Titre>
              <Paragraphe>Pas de dépôt de garantie.</Paragraphe>
            </Cas>
          </BlocCas>
        </Chapitre>
      </Texte>
    </Situation>
  </ListeSituations>
</Publication>
"""


def test_heading_path_is_kept(tmp_path: Path):
    f = tmp_path / "F999.xml"
    f.write_text(PAGE, encoding="utf-8")
    chunks = parse_page(f)
    texts = [c.text for c in chunks]
    # The intro chunk uses the page description
    assert texts[0].startswith("Dépôt de garantie\nUne somme versée")
    # The chapter chunk carries Situation > Chapitre in its header, and inline tags are joined
    assert "Dépôt de garantie > Logement vide > Quel est le montant ?\nAu maximum 1 mois de loyer." in texts
    # The nested case keeps the full path
    case = next(c for c in chunks if c.section.endswith("Loyer payé chaque trimestre"))
    assert case.section == "Logement vide > Quel est le montant ? > Loyer payé chaque trimestre"
    assert case.page_id == "F999" and case.modified == "2026-11-01"


def test_pages_outside_the_selected_topics_are_skipped(tmp_path: Path):
    f = tmp_path / "F999.xml"
    f.write_text(PAGE.replace("Bail d’habitation", "Impôts"), encoding="utf-8")
    assert parse_page(f) is None


def test_split_lines_never_cuts_a_line():
    lines = ["a" * 500, "b" * 500, "c" * 500]
    pieces = split_lines(lines, max_chars=1200)
    assert pieces == [["a" * 500, "b" * 500], ["c" * 500]]
