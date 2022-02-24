from abc import ABC
from pathlib import Path
from typing import Tuple, TypeVar

import pyknp_eventgraph as pyevg
from pyknp import Tag
from pyknp_eventgraph.helper import convert_mrphs_to_surf
from spacy.tokens import Span, Token

from utils.utils import Event, read_seed_lex


class Labeler(ABC):
    T = TypeVar("T")

    def __init__(
        self, data_file: str, seed_lex_file: str, ani_seed_lex_file: str = None
    ):
        self.data_file = Path(data_file)
        self.seed_lex_file = Path(seed_lex_file)
        self.ani_seed_lex_file = Path(ani_seed_lex_file) if ani_seed_lex_file else None

        self.seed_words = read_seed_lex(self.seed_lex_file)
        self.ani_seed_words = (
            read_seed_lex(self.ani_seed_lex_file) if self.ani_seed_lex_file else None
        )

    def __call__(self, event: T) -> Event:
        vol_label, vol_phrase = self._get_vol_label_and_phrase(event)
        ani_label, ani_phrase = self._get_ani_label_and_phrase(event)
        event2 = self._to_event(event)
        event2.vol_label = vol_label
        event2.vol_phrase = vol_phrase
        event2.ani_label = ani_label
        event2.ani_phrase = ani_phrase
        return event2

    def _to_event(self, event: T) -> Event:
        raise NotImplementedError

    def _get_vol_label_and_phrase(self, event: T) -> Tuple[int, str]:
        raise NotImplementedError

    def _get_ani_label_and_phrase(self, event: T) -> Tuple[int, str]:
        raise NotImplementedError


class JapaneseLabeler(Labeler):
    def _to_event(self, event: pyevg.Event) -> Event:
        tokenized_text = event._to_text(exclude_omission=True, include_modifiers=True)
        text = convert_mrphs_to_surf(tokenized_text)
        return Event(
            path=str(self.data_file.absolute()),
            text=text,
            tokenized_text=tokenized_text,
            full_text=event.sentence.surf,
            tokenized_full_text=event.sentence.mrphs,
            pred_phrase=event.pas.predicate.head.midasi,
            pred_rep=event.pas.predicate.standard_reps,
        )

    def _get_vol_label_and_phrase(self, event: pyevg.Event) -> Tuple[int, str]:
        for c in event.pas.predicate.head.children:  # type: Tag
            mod_str = ("".join([t.midasi for t in c.children]) + c.midasi).strip("、")
            for seed_word in self.seed_words:
                if seed_word.text == mod_str:
                    return seed_word.vol_label, seed_word.text
        return -100, ""

    def _get_ani_label_and_phrase(self, event: pyevg.Event) -> Tuple[int, str]:
        noms = event.pas.arguments.get("ガ２", event.pas.arguments.get("ガ", []))
        if len(noms) == 0:
            return -100, ""

        nom = noms[0]

        if nom.flag == "E":
            if any(exo in nom.surf for exo in ("著者", "読者", "不特定:人")):
                return 1, nom.mrphs
            else:
                return 0, nom.mrphs

        if len(nom.adnominal_events) > 0:
            return -100, ""

        for complement in filter(lambda r: r.label == "補文", event.incoming_relations):
            if complement.head_tid == nom.tag.tag_id - 1:
                return -100, ""

        nom_mrphs = nom.mrphs
        for child in nom.children:
            nom_mrphs = child["mrphs"] + " " + nom_mrphs

        if nom.tag.features.get("SM-人", False) or nom.tag.features.get("SM-組織", False):
            return 1, nom_mrphs
        else:
            return 0, nom_mrphs


class EnglishLabeler(Labeler):
    def _to_event(self, event: Span) -> Event:
        return Event(
            path=str(self.data_file.absolute()),
            text=event.text,
            tokenized_text=event.text,
            full_text=event.sent.doc.text,
            tokenized_full_text=event.sent.doc.text,
            pred_phrase=event.root.text,
            pred_rep=event.root.lemma_,
        )

    def _get_vol_label_and_phrase(self, event: Span) -> Tuple[int, str]:
        for c in event.root.children:  # type: Token
            for seed_word in self.seed_words:
                if c.lemma_ == seed_word.text:
                    return seed_word.vol_label, seed_word.text
        return -100, ""

    def _get_ani_label_and_phrase(self, event: Span) -> Tuple[int, str]:
        # https://nlp.stanford.edu/pubs/bowman-chopra-srw-preprint.pdf
        nsubj = None
        for c in event.root.children:  # type: Token
            if c.dep_ == "nsubj":
                nsubj = c
                break
        if nsubj is None:
            for c in event.root.children:  # type: Token
                if c.dep_ == "csubj":
                    nsubj = c
                    break

        if nsubj is None:
            return -100, ""

        if nsubj.ent_type_ in {"PERSON", "NORP", "ORG"}:
            return 1, nsubj.text
        elif nsubj.ent_type_ != "":
            return 0, nsubj.text

        if nsubj.lemma_ in {"I", "you", "we", "he", "she"}:
            return 1, nsubj.text
        elif nsubj.lemma_ in {"this", "that", "it"}:
            return 0, nsubj.text

        for ani_seed_word in self.ani_seed_words:
            if ani_seed_word.text.lower() == nsubj.lemma_.lower():
                return ani_seed_word.vol_label, nsubj.text

        return -100, ""
